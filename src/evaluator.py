evaluator_prompt = """
You are an expert evaluator of teaching explanations.

Context / Expected Answer:
{expected_answer}

Teacher Explanation:
"{teacher_explanation}"

Student Understanding (optional context):
"{student_reflection}"  # Can be passed if needed

Your tasks:
1. Compare the teacher explanation against the expected answer.
2. Decide correctness:
   - correct → fully matches expected answer
   - partial → mostly correct but missing key points
   - incorrect → wrong or misleading explanation
3. List any missing points.
4. Write a concise feedback comment for the student, highlighting what was good and what could be improved.

Respond ONLY in valid JSON that matches this schema:

{
  "correctness": "correct|partial|incorrect",
  "missing_points": ["missing point 1", "missing point 2"],
  "feedback": "Concise feedback comment"
}

"""
