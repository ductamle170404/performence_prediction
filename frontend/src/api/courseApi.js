import { get } from "./httpClient.js";

export const courseApi = {
  getStudentCourses: (studentId) => get(`/api/students/${studentId}/courses`),
  getAssessments: (code, presentation) =>
    get(`/api/courses/${code}/${presentation}/assessments`),
};
