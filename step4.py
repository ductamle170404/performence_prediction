import pandas as pd

names = ["courses","assessments","vle","studentInfo",
         "studentRegistration","studentAssessment","studentVle"]
dfs = {n: pd.read_csv(f"data/{n}.csv") for n in names}   # sửa đường dẫn nếu cần

# def describe_table(name, d):
#     print(f"\n### {name}  ({d.shape[0]:,} dòng, {d.shape[1]} cột)\n")
#     print("| cột | dtype | null | null % | unique | min | max | ví dụ |")
#     print("|---|---|---|---|---|---|---|---|")
#     for c in d.columns:
#         s = d[c]
#         is_num = pd.api.types.is_numeric_dtype(s)
#         mn = s.min() if is_num else ""
#         mx = s.max() if is_num else ""
#         ex = ", ".join(map(str, s.dropna().unique()[:3]))
#         print(f"| {c} | {s.dtype} | {s.isna().sum():,} | "
#               f"{s.isna().mean()*100:.1f} | {s.nunique():,} | {mn} | {mx} | {ex} |")

# for n, d in dfs.items():
#     describe_table(n, d)

si, sr = dfs["studentInfo"], dfs["studentRegistration"]
sa, a, v, sv = dfs["studentAssessment"], dfs["assessments"], dfs["vle"], dfs["studentVle"]
c = dfs["courses"]
key3 = ["code_module","code_presentation","id_student"]
kmp  = ["code_module","code_presentation"]

print("== A. '?' của assessments.date theo loại bài")
print(a.assign(q=a["date"].astype(str).eq("?")).groupby("assessment_type")["q"].agg(["sum","count"]))

print("\n== B. final_result vs có ngày hủy đăng ký")
sr2 = sr.assign(has_unreg=~sr["date_unregistration"].astype(str).eq("?"))
print(pd.crosstab(si.merge(sr2[key3+["has_unreg"]], on=key3)["final_result"],
                  si.merge(sr2[key3+["has_unreg"]], on=key3)["has_unreg"]))

print("\n== C. Bài đánh giá không có bài nộp")
missing = a[~a["id_assessment"].isin(sa["id_assessment"])]
print(missing.groupby(kmp+["assessment_type"]).size())

print("\n== D. Assessment của GGG")
print(a[a["code_module"]=="GGG"].groupby(["code_presentation","assessment_type"])["weight"].agg(["count","sum"]))

print("\n== E. Mẫu dòng trùng khóa trong studentVle (khác sum_click)")
k = key3+["id_site","date"]
g = sv.groupby(k).sum_click.agg(["count","nunique"])
print("nhóm trùng khóa:", (g["count"]>1).sum(), "| trong đó sum_click khác nhau:", (g["nunique"]>1).sum())

print("\n== F. Sinh viên 2014J đã xuất hiện ở các đợt trước")
tr = si[si["code_presentation"]!="2014J"]["id_student"].unique()
te = si[si["code_presentation"]=="2014J"]
print("dòng test có sinh viên đã thấy ở train:", te["id_student"].isin(tr).sum(), "/", len(te))

print("\n== G. Sinh viên thiếu log / thiếu điểm, theo final_result")
si["has_log"] = si.set_index(key3).index.isin(sv[key3].drop_duplicates().set_index(key3).index)
sa_k = sa.merge(a[["id_assessment"]+kmp], on="id_assessment")[key3].drop_duplicates()
si["has_score"] = si.set_index(key3).index.isin(sa_k.set_index(key3).index)
print(si.groupby("final_result")[["has_log","has_score"]].mean().round(3))

print("\n== H. score '?' và is_banked")
print(sa.assign(q=sa["score"].astype(str).eq("?")).groupby("is_banked")["q"].agg(["sum","count"]))