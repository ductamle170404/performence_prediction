import { get } from "./httpClient.js";

export const instructorApi = {
  getTeachingCourses: (instructorId) => get(`/api/instructors/${instructorId}/courses`),
  getClassStudents: (instructorId, code, presentation, classId) =>
    get(`/api/courses/${code}/${presentation}/classes/${classId}/students`, {
      instructor_id: instructorId,
    }),
  getDemoUsers: () => get("/api/demo/users"),
};
