si, sr = dfs["studentInfo"], dfs["studentRegistration"]
sa, a  = dfs["studentAssessment"], dfs["assessments"]
sv, v  = dfs["studentVle"], dfs["vle"]
key3 = ["code_module","code_presentation","id_student"]

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