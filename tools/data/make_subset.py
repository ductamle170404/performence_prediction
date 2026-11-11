"""Tạo subset ngẫu nhiên (phân tầng) ở mức Student–Course từ OULAD.

Chạy:
    python tools\\data\\make_subset.py --frac 0.2 --seed 42 --src data_pure --out data_subset
Mỗi seed khác nhau -> một subset khác nhau (dùng để lặp thí nghiệm / kiểm tra độ ổn định).
Lấy mẫu phân tầng theo (code_presentation, final_result) để giữ tỉ lệ nhãn và đủ cả 4 đợt học.
"""
import argparse, os
import pandas as pd

KEY3 = ["code_module", "code_presentation", "id_student"]
KMP = ["code_module", "code_presentation"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data_pure", help="thư mục chứa dữ liệu OULAD gốc")
    ap.add_argument("--out", default="data_subset")
    ap.add_argument("--frac", type=float, default=0.2, help="tỉ lệ dòng Student–Course giữ lại")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    si = pd.read_csv(f"{a.src}/studentInfo.csv")
    sampled = (si.groupby(["code_presentation", "final_result"], group_keys=False)
                 .sample(frac=a.frac, random_state=a.seed))
    keys = sampled[KEY3]
    print(f"studentInfo: {len(si):,} -> {len(sampled):,} dòng (frac={a.frac}, seed={a.seed})")

    os.makedirs(a.out, exist_ok=True)
    sampled.to_csv(f"{a.out}/studentInfo.csv", index=False)

    # bảng theo Student–Course: lọc theo khóa
    for name in ["studentRegistration", "studentVle"]:
        d = pd.read_csv(f"{a.src}/{name}.csv")
        d = d.merge(keys, on=KEY3, how="inner")
        d.to_csv(f"{a.out}/{name}.csv", index=False)
        print(f"{name}: {len(d):,} dòng")

    # studentAssessment không có code_module/presentation -> đi qua assessments
    asm = pd.read_csv(f"{a.src}/assessments.csv")
    sa = pd.read_csv(f"{a.src}/studentAssessment.csv")
    sa = sa.merge(asm[["id_assessment"] + KMP], on="id_assessment")
    sa = sa.merge(keys, on=KEY3, how="inner").drop(columns=KMP)
    sa.to_csv(f"{a.out}/studentAssessment.csv", index=False)
    print(f"studentAssessment: {len(sa):,} dòng")

    # bảng danh mục: giữ nguyên
    for name in ["courses", "assessments", "vle"]:
        pd.read_csv(f"{a.src}/{name}.csv").to_csv(f"{a.out}/{name}.csv", index=False)
    print("Xong ->", a.out)


if __name__ == "__main__":
    main()
