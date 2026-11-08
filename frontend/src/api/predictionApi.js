import { get } from "./httpClient.js";

const studentPath = (studentId, code, presentation) =>
  `/api/students/${studentId}/courses/${code}/${presentation}`;
const classPath = (code, presentation, classId) =>
  `/api/courses/${code}/${presentation}/classes/${classId}`;

export const predictionApi = {
  getStudentPrediction: (studentId, code, presentation, cutoffDate) =>
    get(`${studentPath(studentId, code, presentation)}/prediction`, { cutoff_date: cutoffDate }),
  getStudentPredictionHistory: (studentId, code, presentation) =>
    get(`${studentPath(studentId, code, presentation)}/prediction-history`),

  // Các API của giảng viên cần instructorId để backend kiểm tra phân công.
  getCourseRisk: (instructorId, code, presentation) =>
    get(`/api/courses/${code}/${presentation}/risk`, { instructor_id: instructorId }),
  getClassRisk: (instructorId, code, presentation, classId) =>
    get(`${classPath(code, presentation, classId)}/risk`, { instructor_id: instructorId }),
  getRiskHistory: (instructorId, code, presentation, classId) =>
    get(`${classPath(code, presentation, classId)}/risk-history`, { instructor_id: instructorId }),
};
