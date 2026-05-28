"""2b. 对 NCBI 无蛋白注释的 acc，下 .fna + pyrodigal 现做蛋白质（CPU；手动跑）。

承接 2.download_faa.py 的 logs/missing_protein.txt（~21,542 个 acc：~21,503 GenBank
无注释 MAG + ~39 RefSeq 极少数未跑 PGAP）。流程：

  NCBI Datasets API (GENOME_FASTA) → .fna in-memory → pyrodigal(meta) → .faa.gz 落盘

设计要点：
- **不落 .fna 中间文件**：pyrodigal 直接读 in-memory bytes 并写 gzip 翻译，省 ~65GB 临时空间。
- **断点续**：faa/<acc>.faa.gz 已存在跳过；旧 log 里的 no_fna / zero / error 也合并跳过。
- **meta 模式**：assemble 质量参差（多 MAG），meta=True 无需训练、对单 contig 友好，
  与 prodigal v2.6.3+ 算法等价（pyrodigal 已严格验证 vs upstream）。
- **多线程**：下载 IO 是瓶颈，pyrodigal SIMD + GIL release 使得 ThreadPool 即可吃满 CPU。
- 仅依赖标准库 + pyrodigal。运行 env: caiqy_bacformer_prior。

用法（从 MCFProjet 根目录）：
    python bacformer_prior/scripts/2b.prodigal.py                # 全量
    python bacformer_prior/scripts/2b.prodigal.py --workers 12   # 调并行
    python bacformer_prior/scripts/2b.prodigal.py --limit 50     # pilot 试跑
"""
import argparse
import gzip
import io
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyrodigal

ROOT = "/home/cml_lab/caiqy/project/MCFProjet"
DATA = f"{ROOT}/data/bacformer_prior"
FAA_DIR = f"{DATA}/faa"
LOG_DIR = f"{DATA}/logs"
# GENOME_FASTA = .fna（核酸基因组），与 2.download_faa.py 的 PROT_FASTA 平行
API = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/accession/{}/download?include_annotation_type=GENOME_FASTA"


def to_ncbi(acc: str) -> str:
    """RS_GCF_/GB_GCA_ -> GCF_/GCA_（NCBI assembly accession）。"""
    return acc.split("_", 1)[1] if acc[:3] in ("RS_", "GB_") else acc


def parse_fna(blob: bytes) -> list[tuple[str, bytes]]:
    """In-memory FASTA 解析为 [(contig_id, seq_bytes), ...]，不落盘。"""
    contigs = []
    cur_id, cur_seq = None, []
    for line in blob.splitlines():
        if line.startswith(b">"):
            if cur_id is not None:
                contigs.append((cur_id, b"".join(cur_seq)))
            cur_id = line[1:].split(maxsplit=1)[0].decode(errors="replace")
            cur_seq = []
        else:
            cur_seq.append(line.strip())
    if cur_id is not None:
        contigs.append((cur_id, b"".join(cur_seq)))
    return contigs


def process_one(acc: str, retries: int = 4) -> tuple[str, int]:
    """返回 (状态, 蛋白数)。状态: 'ok'/'skip'/'no_fna'/'zero'/'error'。"""
    out = f"{FAA_DIR}/{acc}.faa.gz"
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return "skip", 0
    url = API.format(to_ncbi(acc))
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "bacformer_prior/2b"})
            data = urllib.request.urlopen(req, timeout=180).read()
            zf = zipfile.ZipFile(io.BytesIO(data))
            fna_names = [n for n in zf.namelist() if n.endswith(".fna") or n.endswith("_genomic.fna")]
            if not fna_names:
                return "no_fna", 0
            raw = zf.read(fna_names[0])
            if not raw:
                return "no_fna", 0
            contigs = parse_fna(raw)
            if not contigs:
                return "no_fna", 0
            # 每个 acc 自己一个 GeneFinder（meta=True 无状态，但 Genes 对象按 contig 累积，
            # 复用同一 finder 跑多 contig 后通过 write_translations 写出每个 contig 的蛋白）
            finder = pyrodigal.GeneFinder(meta=True)
            n_proteins = 0
            tmp = out + ".tmp"
            with gzip.open(tmp, "wt") as g:
                for cid, seq in contigs:
                    if len(seq) < 60:  # 太短 contig 没意义
                        continue
                    genes = finder.find_genes(bytes(seq))
                    n_proteins += len(genes)
                    if len(genes):
                        genes.write_translations(g, cid)
            if n_proteins == 0:
                os.remove(tmp)
                return "zero", 0
            os.replace(tmp, out)
            return "ok", n_proteins
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)  # 退避
                continue
            return "error", 0
        except Exception:
            time.sleep(2 ** attempt)
    return "error", 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8, help="并行线程数（CPU 24 核，建议 8-16）")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只处理前 N 个（pilot）")
    ap.add_argument("--input", default=f"{LOG_DIR}/missing_protein.txt")
    args = ap.parse_args()

    os.makedirs(FAA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    with open(args.input) as f:
        accs = sorted({l.strip() for l in f if l.strip()})

    # 断点续：旧 log 里的 no_fna/zero/error 也直接跳过，避免重打 API
    skip_set = set()
    for fn in ("prodigal_no_fna.txt", "prodigal_zero.txt"):
        path = f"{LOG_DIR}/{fn}"
        if os.path.exists(path):
            with open(path) as f:
                skip_set |= {l.strip() for l in f if l.strip()}
    accs_todo = [a for a in accs if a not in skip_set]
    print(f"待处理 {len(accs_todo):,} 个 acc（输入 {len(accs):,}，"
          f"旧 no_fna/zero log 跳 {len(accs) - len(accs_todo):,}；"
          f"已落 .faa.gz 也跳过）；workers={args.workers}")

    if args.limit:
        accs_todo = accs_todo[: args.limit]
        print(f"limit={args.limit} → 试跑前 {len(accs_todo)} 个")

    counts = {"ok": 0, "skip": 0, "no_fna": 0, "zero": 0, "error": 0}
    new_no_fna, new_zero, new_errors = [], [], []
    t0 = time.time()
    total_proteins = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_one, a): a for a in accs_todo}
        for i, fut in enumerate(as_completed(futs), 1):
            acc = futs[fut]
            st, np_ = fut.result()
            counts[st] += 1
            total_proteins += np_
            if st == "no_fna":
                new_no_fna.append(acc)
            elif st == "zero":
                new_zero.append(acc)
            elif st == "error":
                new_errors.append(acc)
            if i % 100 == 0 or i == len(accs_todo):
                el = time.time() - t0
                rate = i / max(el, 1e-6)
                eta = (len(accs_todo) - i) / max(rate, 1e-6)
                avg_prot = total_proteins / max(counts["ok"], 1)
                print(f"  [{i:,}/{len(accs_todo):,} {i/len(accs_todo)*100:4.1f}%] "
                      f"ok={counts['ok']} skip={counts['skip']} no_fna={counts['no_fna']} "
                      f"zero={counts['zero']} error={counts['error']} "
                      f"| {rate:.1f}/s avg_prot={avg_prot:.0f} 已用{el/60:.0f}min ETA{eta/60:.0f}min",
                      flush=True)

    # 合并旧 + 新（避免覆盖丢历史）
    def merge_write(fn: str, new_list: list) -> int:
        path = f"{LOG_DIR}/{fn}"
        old = set()
        if os.path.exists(path):
            with open(path) as f:
                old = {l.strip() for l in f if l.strip()}
        merged = sorted(old | set(new_list))
        with open(path, "w") as f:
            f.write("\n".join(merged) + ("\n" if merged else ""))
        return len(merged)

    n_nf = merge_write("prodigal_no_fna.txt", new_no_fna)
    n_zr = merge_write("prodigal_zero.txt", new_zero)
    # error 全量覆盖（这次仍败的才留下，断点续再补）
    with open(f"{LOG_DIR}/prodigal_errors.txt", "w") as f:
        f.write("\n".join(new_errors) + ("\n" if new_errors else ""))

    print(f"\n完成：{counts}")
    print(f"  新增蛋白 ≈ {total_proteins:,}（avg {total_proteins/max(counts['ok'],1):.0f}/genome）")
    print(f"  no_fna 累计 {n_nf}（NCBI 端真无 .fna，丢弃，无法补） → logs/prodigal_no_fna.txt")
    print(f"  zero  累计 {n_zr}（pyrodigal 跑出 0 蛋白，损坏/极小 assembly） → logs/prodigal_zero.txt")
    print(f"  error 本轮 {len(new_errors)}（断点续可重跑补） → logs/prodigal_errors.txt")
    if new_errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
