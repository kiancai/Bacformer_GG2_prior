"""2. 下载基因组蛋白序列 .faa（手动跑；CPU/IO，无 GPU，无需装包）。

读 genus_to_genomes.tsv 的**唯一** accession（~62k，去重；多个 GG2 token 共享同一代表基因组），
逐个从 NCBI Datasets API 取 PROT_FASTA，gzip 存到 data/bacformer_prior/faa/<accession>.faa.gz。

- **断点续传**：已存在的 .faa.gz 跳过 → 中断后重跑即可。
- **并行 + 重试**：ThreadPoolExecutor + 429/5xx 退避。
- **记录缺注释**：NCBI 无现成蛋白的（多为未注释 GenBank）写入 logs/missing_protein.txt
  → 留给 2b（下 .fna + Prodigal）。本脚本**不做注释**。
- 仅用标准库，任何 python3 可跑。

用法（从 MCFProjet 根目录）：
    python bacformer_prior/scripts/2.download_faa.py                # 全量
    python bacformer_prior/scripts/2.download_faa.py --workers 8    # 调并行(被限流就调小)
    python bacformer_prior/scripts/2.download_faa.py --limit 50     # 先小批试跑

【更快的替代：NCBI datasets CLI 批量】本脚本会同时导出 logs/accessions.txt；若已装 datasets 二进制可：
    datasets download genome accession --inputfile data/bacformer_prior/logs/accessions.txt \
        --include protein --dehydrated --filename faa.zip   # 再 rehydrate
"""
import argparse
import csv
import gzip
import io
import os
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = "/home/cml_lab/caiqy/project/MCFProjet"
DATA = f"{ROOT}/data/bacformer_prior"
MAPPING = f"{DATA}/genus_to_genomes.tsv"
FAA_DIR = f"{DATA}/faa"
LOG_DIR = f"{DATA}/logs"
API = "https://api.ncbi.nlm.nih.gov/datasets/v2alpha/genome/accession/{}/download?include_annotation_type=PROT_FASTA"


def to_ncbi(acc: str) -> str:
    """RS_GCF_.../GB_GCA_... -> GCF_.../GCA_...（NCBI assembly accession）。"""
    return acc.split("_", 1)[1] if acc[:3] in ("RS_", "GB_") else acc


def load_accessions(path: str) -> list[str]:
    accs = set()
    with open(path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["accession"]:
                accs.add(row["accession"])
    return sorted(accs)


def fetch_one(acc: str, known_missing: set, retries: int = 4) -> str:
    """返回状态：'ok' / 'skip' / 'skip_missing' / 'missing'(无蛋白) / 'error'。

    断点续：
    - faa/<acc>.faa.gz 已落 → 'skip'（不打 API）。
    - acc 在 known_missing（上次跑发现 NCBI 无注释）→ 'skip_missing'（不打 API）。
    """
    out = f"{FAA_DIR}/{acc}.faa.gz"
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return "skip"
    if acc in known_missing:
        return "skip_missing"
    url = API.format(to_ncbi(acc))
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "bacformer_prior/1.0"})
            data = urllib.request.urlopen(req, timeout=120).read()
            zf = zipfile.ZipFile(io.BytesIO(data))
            faa = [n for n in zf.namelist() if n.endswith("protein.faa")]
            if not faa:
                return "missing"  # NCBI 无现成蛋白注释（→ Prodigal）
            raw = zf.read(faa[0])
            if not raw:
                return "missing"
            tmp = out + ".tmp"
            with gzip.open(tmp, "wb") as g:
                g.write(raw)
            os.replace(tmp, out)
            return "ok"
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)  # 退避
                continue
            return "error"
        except Exception:
            time.sleep(2 ** attempt)
    return "error"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8, help="并行线程数（被 NCBI 限流就调小，如 4）")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只下前 N 个（试跑）")
    ap.add_argument("--input", default=MAPPING)
    args = ap.parse_args()

    os.makedirs(FAA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    accs = load_accessions(args.input)
    with open(f"{LOG_DIR}/accessions.txt", "w") as f:
        f.write("\n".join(to_ncbi(a) for a in accs) + "\n")
    if args.limit:
        accs = accs[: args.limit]

    # 断点续：载入上次跑的 missing_protein.txt（NCBI 无注释、不会有 .faa.gz 落盘），
    # 这次直接跳过、不再打 API。结尾合并 (旧 ∪ 新发现) 再写回。
    missing_path = f"{LOG_DIR}/missing_protein.txt"
    known_missing: set = set()
    if os.path.exists(missing_path):
        with open(missing_path) as f:
            known_missing = {l.strip() for l in f if l.strip()}
    print(f"唯一 accession {len(accs):,}（已存的会跳过）；"
          f"已知 missing {len(known_missing):,}（也跳过）；workers={args.workers}")

    counts = {"ok": 0, "skip": 0, "skip_missing": 0, "missing": 0, "error": 0}
    new_missing, errors = [], []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_one, a, known_missing): a for a in accs}
        for i, fut in enumerate(as_completed(futs), 1):
            acc = futs[fut]
            st = fut.result()
            counts[st] += 1
            if st == "missing":
                new_missing.append(acc)
            elif st == "error":
                errors.append(acc)
            if i % 200 == 0 or i == len(accs):
                el = time.time() - t0
                rate = i / max(el, 1e-6)
                eta = (len(accs) - i) / max(rate, 1e-6)
                print(f"  [{i:,}/{len(accs):,} {i/len(accs)*100:4.1f}%] "
                      f"ok={counts['ok']} skip={counts['skip']} skip_missing={counts['skip_missing']} "
                      f"missing={counts['missing']} error={counts['error']} "
                      f"| {rate:.1f}/s 已用{el/60:.0f}min ETA{eta/60:.0f}min", flush=True)

    # missing 合并旧 + 这次新发现；error 全量覆盖（这次仍败的才留下，断点续再补）
    all_missing = sorted(known_missing | set(new_missing))
    with open(missing_path, "w") as f:
        f.write("\n".join(all_missing) + ("\n" if all_missing else ""))
    with open(f"{LOG_DIR}/download_errors.txt", "w") as f:
        f.write("\n".join(errors) + ("\n" if errors else ""))
    print(f"\n完成：{counts}")
    print(f"  无现成蛋白合并(→Prodigal/2b)：{len(all_missing)} (新增 {len(new_missing)}) → logs/missing_protein.txt")
    print(f"  报错(可重跑本脚本断点续)：{len(errors)} → logs/download_errors.txt")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
