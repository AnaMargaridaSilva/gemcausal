{
    "task_description": "A causal relationship refers to a connection in which one event causes another, indicating that the two events are interlinked such that one influences the other.\n Please perform the Causal Span Detection task based on the following rules, ensuring no irrelevant outputs:\n1. Please insert <c> before the word where the cause span begins and </c> after the word where the span ends.\n2. Please insert <e> before the word where the effect span begins and </e> after the word where the span ends.\n3. If there are multiple pairs of cause and effect, please use the following format:\nRelation1: [text with tags wraping the first cause and effect]\nRelation2: [text with tags wraping the second cause and effect]\nRelation3: [text with tags wraping the third cause and effect]\n\n",
    "header_example": "Annotation Examples:\n",
    "format_text": "Text: {}\n",
    "format_class": "Text with tags: \n{}\n",
    "question": "According to the policy above, please add the appropriate tags in the following text.\n"
}
