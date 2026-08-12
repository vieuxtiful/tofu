# Natural Language Processing with Deep Learning
## CS224N/Ling284 - Lecture 1: Introduction

**Instructors:** Christopher Manning and Richard Socher

---

## Lecture Plan
1. **What is Natural Language Processing?** The nature of human language (15 mins)
2. **What is Deep Learning?** (15 mins)
3. **Course logistics** (10 mins)
4. **Why is language understanding difficult** (10 mins)
5. **Intro to the application of Deep Learning to NLP** (25 mins)

*Emergency time reserves: 5 mins*

---

## 1. What is Natural Language Processing (NLP)?
*   NLP is a field at the intersection of:
    *   Computer science
    *   Artificial intelligence
    *   Linguistics
*   **Goal:** For computers to process or "understand" natural language in order to perform useful tasks.
    *   **Performing Tasks:** e.g., making appointments, buying things.
    *   **Question Answering:** e.g., Siri, Google Assistant, Facebook M, Cortana.
*   Fully understanding and representing the meaning of language is a difficult goal.
    *   Perfect language understanding is **AI-complete**.

### NLP Applications
Applications range from simple to complex:
*   Spell checking, keyword search, finding synonyms.
*   Extracting information from websites (prices, dates, locations, names).
*   Classifying: reading level, sentiment analysis.
*   Machine translation.
*   Spoken dialog systems.
*   Complex question answering.

### NLP in Industry
NLP is taking off in various sectors:
*   Search (written and spoken).
*   Online advertisement matching.
*   Automated/assisted translation.
*   Sentiment analysis for marketing or finance/trading.
*   Speech recognition.
*   Chatbots / Dialog agents (customer support, controlling devices, ordering goods).

---

## What’s special about human language?
*   **Deliberate Communication:** A system specifically constructed to convey meaning; not just an environmental signal.
*   **Quick Learning:** Kids learn it amazingly fast.
*   **Discrete/Symbolic System:** Uses categorical symbols (e.g., rocket = 🚀, violin = 🎻).
*   **Encoding Invariance:** Symbols are invariant across different encodings (sound, gesture, images/writing).
*   **Continuous Signal:** While symbolic, brain encoding appears as continuous patterns of activation, and symbols are transmitted via continuous signals.
*   **Sparsity:** The large vocabulary and symbolic encoding create a sparsity problem for machine learning.

---

## 2. What’s Deep Learning (DL)?
*   Deep learning is a subfield of machine learning.
*   **Traditional Machine Learning:**
    *   Works well because of human-designed representations and input features.
    *   Example: Features for finding named entities (locations, organizations).
    *   ML becomes an optimization of weights to make a final prediction.
*   **Deep Learning (Representation Learning):**
    *   Attempts to automatically learn good features or representations.
    *   Attempts to learn multiple levels of representation and an output from "raw" inputs (sound, characters, words).

### History and Terminology
*   Focuses on **neural networks**, the dominant model family in DL.
*   Some see it as "clever terminology for stacked logistic regression units," but it involves interesting modeling principles (end-to-end) and connections to neuroscience.
*   For a long history, see: *Deep Learning in Neural Networks: An Overview* by Jürgen Schmidhuber.

### Reasons for Exploring Deep Learning
*   **Manual features** are often over-specified, incomplete, and time-consuming to design.
*   **Learned features** are easy to adapt and fast to learn.
*   Provides a flexible, universal framework for representing world, visual, and linguistic information.
*   Can learn **unsupervised** (raw text) and **supervised** (labeled data).

### Why this decade?
*   Large amounts of training data.
*   Faster machines (multicore CPU/GPUs).
*   New models, algorithms, and ideas (intermediate representations, end-to-end learning, transfer learning).

---

## 3. Course Logistics
*   **Instructors:** Christopher Manning & Richard Socher
*   **TAs:** Many wonderful people!
*   **Time:** TuTh 4:30–5:50, Nvidia Aud
*   **Webpage:** [http://cs224n.stanford.edu/](http://cs224n.stanford.edu/)

### Prerequisites
*   Proficiency in **Python**.
*   Multivariate Calculus, Linear Algebra.
*   Basic Probability and Statistics.
*   Fundamentals of Machine Learning (loss functions, derivatives, gradient descent).

### Learning Objectives
1.  Modern methods for deep learning (Recurrent networks, attention, etc.).
2.  Big picture understanding of human languages and their difficulties.
3.  Ability to build systems for major NLP problems (word similarity, parsing, MT, NER, QA).

### Grading Policy
*   3 Assignments: 51% (17% each)
*   Midterm Exam: 17%
*   Final Project or Assignment 4: 30%
*   Final Poster Session: 2%

---

## 4. Why is NLP hard?
*   Complexity in representing and using knowledge (linguistic, situational, world, visual).
*   **Ambiguity:** Human languages are highly ambiguous compared to formal languages.
*   Interpretation depends on real-world, common sense, and contextual knowledge.
*   *Examples of ambiguous headlines:*
    1.  "The Pope’s baby steps on gays"
    2.  "Boy paralyzed after tumor fights back to gain black belt"
    3.  "Scientists study whales from space"
    4.  "Juvenile Court to Try Shooting Defendant"

---

## 5. Deep NLP = Deep Learning + NLP
Combining NLP ideas with representation learning and DL methods.

### Word Meaning
*   Represented as a **neural word vector** (e.g., a vector of numbers).
*   **Word Similarities:** Nearest words to "frog" include frogs, toad, litoria, etc.

### Representations of NLP Levels
*   **Morphology:** Every morpheme is a vector; a neural network combines vectors.
*   **Parsing:** Neural networks can accurately determine sentence structure.
*   **Semantics:** Every word and phrase is a vector; a neural network combines them without intermediate artificial logical languages.
*   **Sentiment Analysis:** Uses Recursive Neural Networks (RecursiveNN) instead of curated dictionaries.
*   **Question Answering:** Facts are stored as vectors.
*   **Dialogue Agents:** Use Neural Language Models (Recurrent Neural Networks).
*   **Machine Translation:** Source sentences are mapped to vectors, then output sentences are generated (Neural Machine Translation).

---

## Conclusion
*   We will study how to learn vector representations for words in the next lecture.
*   Next week: How neural networks work and how they use these vectors for all NLP levels.
