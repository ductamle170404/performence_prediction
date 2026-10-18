import pandas as pd

names = ["courses","assessments","vle","studentInfo",
         "studentRegistration","studentAssessment","studentVle"]
dfs = {n: pd.read_csv(f"data/{n}.csv") for n in names}   # sửa đường dẫn nếu cần

def describe_table(name, d):
    print(f"\n### {name}  ({d.shape[0]:,} dòng, {d.shape[1]} cột)\n")
    print("| cột | dtype | null | null % | unique | min | max | ví dụ |")
    print("|---|---|---|---|---|---|---|---|")
    for c in d.columns:
        s = d[c]
        is_num = pd.api.types.is_numeric_dtype(s)
        mn = s.min() if is_num else ""
        mx = s.max() if is_num else ""
        ex = ", ".join(map(str, s.dropna().unique()[:3]))
        print(f"| {c} | {s.dtype} | {s.isna().sum():,} | "
              f"{s.isna().mean()*100:.1f} | {s.nunique():,} | {mn} | {mx} | {ex} |")

for n, d in dfs.items():
    describe_table(n, d)

si, sr = dfs["studentInfo"], dfs["studentRegistration"]
sa, a, v, sv = dfs["studentAssessment"], dfs["assessments"], dfs["vle"], dfs["studentVle"]
c = dfs["courses"]
key3 = ["code_module","code_presentation","id_student"]
kmp  = ["code_module","code_presentation"]

print("== 1. Số giá trị '?' theo cột (bỏ qua studentVle: toàn cột số)")
for n, d in dfs.items():
    if n == "studentVle": continue
    for col in d.columns:
        if not pd.api.types.is_numeric_dtype(d[col]):
            q = (d[col].astype(str).str.strip() == "?").sum()
            if q: print(f"{n}.{col}: {q:,} ({q/len(d)*100:.1f}%)")

print("\n== 2. studentVle: dòng trùng khóa")
k = key3 + ["id_site","date"]
print("số dòng thuộc nhóm trùng khóa:", sv.duplicated(k, keep=False).sum())
print("số dòng trùng hoàn toàn (cả sum_click):", sv.duplicated().sum())
print(sv[sv.duplicated(k, keep=False)].sort_values(k).head(8))

print("\n== 3. Quan hệ chưa kiểm (0 là tốt)")
print("studentInfo -> courses:", len(si.merge(c, on=kmp, how="left", indicator=True).query("_merge=='left_only'")))
print("assessments -> courses:", len(a.merge(c, on=kmp, how="left", indicator=True).query("_merge=='left_only'")))
print("studentVle -> studentInfo:", len(sv[key3].drop_duplicates().merge(si[key3], on=key3, how="left", indicator=True).query("_merge=='left_only'")))
sa2 = sa.merge(a[["id_assessment"]+kmp], on="id_assessment")
print("studentAssessment -> studentInfo:", len(sa2[key3].drop_duplicates().merge(si[key3], on=key3, how="left", indicator=True).query("_merge=='left_only'")))

print("\n== 4. Sinh viên xuất hiện ở bao nhiêu dòng Student–Course")
print(si.groupby("id_student").size().value_counts().sort_index())

print("\n== 5. Nhãn theo đợt học")
print(pd.crosstab(si["code_presentation"], si["final_result"]))

print("\n== 6. Bài thi (Exam) và ngày bị thiếu")
ex = a[a["assessment_type"]=="Exam"]
print(ex.groupby(kmp)[["date","weight"]].first())

print("\n== 7. Tổng trọng số theo môn/đợt/loại bài")
print(a.groupby(kmp+["assessment_type"])["weight"].sum().unstack())

print("\n== 8. date_submitted vượt độ dài môn")
m = sa2.merge(c, on=kmp)
print("số bài nộp sau khi môn kết thúc:", (m["date_submitted"] > m["module_presentation_length"]).sum())