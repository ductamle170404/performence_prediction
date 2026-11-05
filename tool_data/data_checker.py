import pandas as pd

names = ["courses","assessments","vle","studentInfo",
         "studentRegistration","studentAssessment","studentVle"]
dfs = {n: pd.read_csv(f"data_pure/{n}.csv") for n in names}   # sửa đường dẫn nếu cần

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
class_counts = pd.crosstab(si["code_presentation"], si["final_result"])
semester_totals = class_counts.sum(axis=1)
class_percentages = class_counts.div(semester_totals, axis=0).mul(100)
result_by_class = class_counts.astype(str) + " (" + class_percentages.apply(
    lambda column: column.map(lambda percentage: f"{percentage:.1f}%")
) + ")"
result_by_class["Tổng"] = semester_totals.astype(str) + " (100.0%)"
print(result_by_class)

print("\n== 6. Bài thi (Exam) và ngày bị thiếu")
ex = a[a["assessment_type"]=="Exam"]
print(ex.groupby(kmp)[["date","weight"]].first())

print("\n== 7. Tổng trọng số theo môn/đợt/loại bài")
print(a.groupby(kmp+["assessment_type"])["weight"].sum().unstack())

print("\n== 8. date_submitted vượt độ dài môn")
m = sa2.merge(c, on=kmp)
print("số bài nộp sau khi môn kết thúc:", (m["date_submitted"] > m["module_presentation_length"]).sum())

si, sr = dfs["studentInfo"], dfs["studentRegistration"]
sa, a  = dfs["studentAssessment"], dfs["assessments"]
sv, v  = dfs["studentVle"], dfs["vle"]
key3 = ["code_module","code_presentation","id_student"]

print("\n== 9. Kiểm tra khóa chính và quan hệ giữa các bảng")

print("== Kiểm tra khóa chính (True = đúng là duy nhất)")
print("studentInfo key3:", not si.duplicated(key3).any())
print("studentRegistration key3:", not sr.duplicated(key3).any())
print("studentAssessment (id_assessment,id_student):",
      not sa.duplicated(["id_assessment","id_student"]).any())
print("assessments id_assessment:", a["id_assessment"].is_unique)
print("vle id_site:", v["id_site"].is_unique)
print("studentVle key (6 cột đầu, kèm date):",
      not sv.duplicated(key3+["id_site","date"]).any())

print("\n== Kiểm tra quan hệ (số dòng không khớp, 0 là tốt)")
print("studentAssessment -> assessments:",
      (~sa["id_assessment"].isin(a["id_assessment"])).sum())
print("studentVle -> vle (id_site):",
      (~sv["id_site"].isin(v["id_site"])).sum())
print("studentRegistration -> studentInfo:",
      len(sr.merge(si[key3], on=key3, how="left", indicator=True)
          .query("_merge=='left_only'")))

print("\n== Phân bố nhãn")
print(si["final_result"].value_counts())
print("\n== Số đợt học / môn")
print(dfs["courses"].groupby("code_module")["code_presentation"].apply(list))

print("\n== Loại bài đánh giá")
print(a["assessment_type"].value_counts())
print("\n== Loại hoạt động VLE")
print(v["activity_type"].value_counts())