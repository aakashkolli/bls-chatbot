"""
BLS Chatbot Evaluation — Test Question Bank

All questions from the standardized test document.
Structure preserves original headings and numbering exactly.
"""

QUESTION_SECTIONS = [
    {
        "section": "Prospective Student Questions",
        "subsections": [
            {
                "name": "General",
                "questions": [
                    {"num": 1, "text": "What is the Bachelor of Liberal Studies (BLS) degree?"},
                    {"num": 2, "text": "Who is the BLS program designed for?"},
                    {"num": 3, "text": "Is the BLS degree fully online?"},
                    {"num": 4, "text": "How is BLS different from other LAS majors?"},
                ],
            },
            {
                "name": "Admissions",
                "questions": [
                    {"num": 5,  "text": "What are the admission requirements?"},
                    {"num": 6,  "text": "Can I transfer credits? How many?"},
                    {"num": 7,  "text": "Do you accept community college credits?"},
                    {"num": 8,  "text": "Is there a minimum GPA requirement?"},
                    {"num": 9,  "text": "When are application deadlines?"},
                    {"num": 10, "text": "How do I send transcripts?"},
                ],
            },
            {
                "name": "Academics",
                "questions": [
                    {"num": 11, "text": "What concentrations are available?"},
                    {"num": 12, "text": "Can I design my own area of focus?"},
                    {"num": 13, "text": "How long does it take to complete the degree?"},
                    {"num": 14, "text": "Are there required courses?"},
                    {"num": 15, "text": "Are classes synchronous or asynchronous?"},
                    {"num": 16, "text": "Can I take classes part-time?"},
                    {"num": 17, "text": "How do online classes work?"},
                ],
            },
            {
                "name": "Cost & Financial Aid",
                "questions": [
                    {"num": 18, "text": "What is the tuition cost?"},
                    {"num": 19, "text": "Are there scholarships available?"},
                    {"num": 20, "text": "Is financial aid available for part-time students?"},
                    {"num": 21, "text": "Are there payment plans?"},
                    {"num": 22, "text": "Does the BLS program accept the U of I employee tuition waiver?"},
                ],
            },
            {
                "name": "Career-Oriented Questions",
                "questions": [
                    {"num": 23, "text": "What careers can I pursue with a BLS degree?"},
                    {"num": 24, "text": "Does BLS prepare me for graduate school?"},
                    {"num": 25, "text": "Are internships required?"},
                    {"num": 26, "text": "Are there career services available to BLS students?"},
                    {"num": 27, "text": "Can I network with alumni?"},
                ],
            },
        ],
    },
    {
        "section": "Current Student Questions",
        "subsections": [
            {
                "name": "Advising & Registration",
                "questions": [
                    {"num": 1, "text": "How do I schedule an advising appointment?"},
                    {"num": 2, "text": "Who is my academic advisor?"},
                    {"num": 3, "text": "When can I register for classes?"},
                    {"num": 4, "text": "How do I declare or change my concentration?"},
                ],
            },
            {
                "name": "Degree Progress",
                "questions": [
                    {"num": 5, "text": "How do I check my degree audit?"},
                    {"num": 6, "text": "What courses do I still need to graduate?"},
                    {"num": 7, "text": "Can a previous course count toward my concentration?"},
                ],
            },
            {
                "name": "Policies",
                "questions": [
                    {"num": 8,  "text": "What is the GPA requirement to stay in good standing?"},
                    {"num": 9,  "text": "How do I apply for graduation?"},
                    {"num": 10, "text": "What happens if I need to take a leave of absence?"},
                ],
            },
        ],
    },
    {
        "section": "More Subjective/Open-Ended Questions",
        "subsections": [
            {
                "name": "",
                "questions": [
                    {"num": 1,  "text": "Will this degree feel legitimate compared to traditional majors?"},
                    {"num": 2,  "text": "What kinds of students succeed in BLS?"},
                    {"num": 3,  "text": "Is BLS a good option if I'm returning to school after many years?"},
                    {"num": 4,  "text": "How flexible is the program if I work full-time?"},
                    {"num": 5,  "text": "Will I feel connected to campus as an online student?"},
                    {"num": 6,  "text": "How rigorous are BLS courses?"},
                    {"num": 7,  "text": "Will I be challenged academically?"},
                    {"num": 8,  "text": "Is it possible to build depth in one subject, or is it broad by design?"},
                    {"num": 9,  "text": "How much autonomy do I have in shaping my coursework?"},
                    {"num": 10, "text": "Will I get meaningful interaction with faculty?"},
                    {"num": 11, "text": "Are classes discussion-heavy or more independent?"},
                    {"num": 12, "text": "What is the workload like for someone balancing work and family?"},
                    {"num": 13, "text": "Can BLS help me pivot industries?"},
                    {"num": 14, "text": "Will this degree help me move into leadership roles?"},
                    {"num": 15, "text": "How do employers view interdisciplinary degrees?"},
                    {"num": 16, "text": "Is BLS a good stepping stone to graduate school?"},
                    {"num": 17, "text": "If my goal is a promotion, grad school, or career change, is BLS the best path?"},
                ],
            },
        ],
    },
]


def get_all_questions_flat():
    """Return a flat list of all questions with section/subsection metadata."""
    flat = []
    for section_obj in QUESTION_SECTIONS:
        section = section_obj["section"]
        for sub in section_obj["subsections"]:
            subsection = sub["name"]
            for q in sub["questions"]:
                flat.append(
                    {
                        "section": section,
                        "subsection": subsection,
                        "num": q["num"],
                        "text": q["text"],
                        # Unique global key for checkpointing
                        "key": f"{section}|{subsection}|{q['num']}",
                    }
                )
    return flat


# Representative 12-question subset for prompt-improvement re-runs
SUBSET_KEYS = [
    "Prospective Student Questions|General|1",
    "Prospective Student Questions|General|3",
    "Prospective Student Questions|Admissions|5",
    "Prospective Student Questions|Admissions|8",
    "Prospective Student Questions|Academics|11",
    "Prospective Student Questions|Academics|13",
    "Prospective Student Questions|Cost & Financial Aid|18",
    "Prospective Student Questions|Career-Oriented Questions|23",
    "Current Student Questions|Advising & Registration|1",
    "Current Student Questions|Policies|8",
    "More Subjective/Open-Ended Questions||3",
    "More Subjective/Open-Ended Questions||15",
]


def get_subset_questions():
    """Return the representative 12-question subset."""
    flat = get_all_questions_flat()
    return [q for q in flat if q["key"] in SUBSET_KEYS]


if __name__ == "__main__":
    all_q = get_all_questions_flat()
    print(f"Total questions: {len(all_q)}")
    for q in all_q:
        print(f"  [{q['section']} > {q['subsection']}] Q{q['num']}: {q['text'][:70]}")
