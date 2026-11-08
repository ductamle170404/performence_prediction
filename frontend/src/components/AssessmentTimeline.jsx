export default function AssessmentTimeline({ assessments, currentDay, length }) {
  if (!assessments.length) {
    return <p className="muted">Chưa có hoạt động nào trong học kỳ này.</p>;
  }
  return (
    <ol className="timeline">
      {assessments.map((a) => {
        const passed = currentDay == null || a.date <= currentDay;
        return (
          <li key={a.id_assessment} className={passed ? "timeline-item" : "timeline-item upcoming"}>
            <span className="timeline-day">Ngày {a.date}</span>
            <div>
              <strong>{a.assessment_type}</strong>
              <span className="muted"> · mã {a.id_assessment}</span>
              <div className="muted">Trọng số {a.weight}%</div>
            </div>
          </li>
        );
      })}
      {length != null && <li className="timeline-end muted">Kết thúc học kỳ: ngày {length}</li>}
    </ol>
  );
}
