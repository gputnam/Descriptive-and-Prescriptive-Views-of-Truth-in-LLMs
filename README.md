# Descriptive and Prescriptive Views of Truth in LLMs

This was a project to extend the results of the [Geometry of Truth](https://arxiv.org/abs/2310.06824) paper, which found that LLMs represent the truth of a statement, such as "Chicago is in Canada", as a single direction in the space of its final hidden layer. I explored whether the "descriptive" representation of truth would also be "prescriptive": whether it could be used to steer the model in response to being challenged in its response to a math question. I ran the study with the reasoning model DeepSeek-R1-Distill-Qwen-14B, and used the [MATH](https://github.com/hendrycks/math) dataset of math problems. I built on the [Thought Anchors](https://github.com/interp-reasoning/thought-anchors) paper, which, for the purposes of this study, provided sample correct and incorrect answers to the MATH problems from the DeepSeek model.

I built two versions of a possible truth direction, both leveraging the difference-in-mean from a true and false population of tokens. The first used the same Geometry of Truth true and false statements. Since these are provided as outside text provided to the model (as in a prompt), I refer to it as the "Outside Truth" direction.

![Truth PCA](figures/fig1_truth_pca.png)
FIG 1: Final hidden layer values at the final token of True and False statements, along the 1st and 2nd directions of a principal component analysis of the population.

The second used responses to the MATH problems from the Thought Anchors dataset. I took the reasoning traces of correct answers, and then swapped the final answer with a reply from one of the incorrect reasoning traces. I tried both swapping with an incorrect answer to that same question, or a random incorrect answer. (The later case should be more obviously wrong; for example, it could provide a decimal answer to a question asking for an integer). An example incorrect reasoning trace is
> XXX

![Unsureness PCA](figures/fig2_unsure_pca.png)
FIG 2: Final hidden layer values at the final token of Correct and Incorrect responses (both following a Correct reasoning trace) to MATH problems, along the 1st and 2nd directions of a principal component analysis of the population. The Incorrect responses include both false answers provided by the model to that question, and random Incorrect answers. The random Incorrect answers are further away from the Correct answers in this space.

The Outside Truth and Reasoned Truth directions have a small but non-zero cosine similarity of 0.201. I further visualized the two spaces by plotting the Geometry of Truth statements in the PCA space of the MATH answers, and vice-versa. The two populations also separate fairly cleanly in this space. This demonstrates that the model represents the notions of truth from an outside statement and a reasoned statement distinctly, but with some overlap.

![Truth data in unsureness PCA](figures/fig3_truth_in_unsure_pca.png)
![Unsure-swap data in truth PCA](figures/fig4_unsure_in_truth_pca.png)
FIG 3: Distribution of True and False statements from the Geometry of Truth dataset in the PCA space of the MATH questions, and vice-versa.

Next, I tested if I could use either of these directions to steer the model when asked to reevaluate its answers to the MATH questions. To do this, I took correct and incorrect replies from the Thought Anchors dataset, and appended "Are you sure?". I then fed the model forward, steering its final hidden layer as
```
h_t  <-  h_t + r * mean(|h|) * v
```
where `r` is the steering strength, `mean(|h|)` is the mean magnitude of the layer (found to be 335.66), and v is the (unit-normalized) steering vector.

Steering with the Outside Truth direction did very little to change its output. However, steering negatively with the Reasoned Truth direction change the result strikingly. The model would endlessly prevaricate, whether or not it started with the correct answer. Here is one example, with strong steering (`r = -1`):
> XXX

To quantify this pattern, I show the fraction of completed model responses (within a 4096 token cutoff), as well as the fraction of uses of the word "wait" per 1000 words. Steering with the Outside Truth direction has little impact, but steering with the Reasoned Truth direction is quite striking. 

![Completed, correct base](figures/fig5c_completed_vs_r_correct.png)
![Completed, incorrect base](figures/fig5i_completed_vs_r_incorrect.png)
FIG 4: Fraction of completed responses (within as 4096 token limit), for steering in the Outside Truth (FIG 1) and Reasoned Truth (FIG 2) directions.

![Wait rate, correct base](figures/fig6c_wait_vs_r_correct.png)
![Wait rate, incorrect base](figures/fig6i_wait_vs_r_incorrect.png)
FIG 5: Uses of the word "Wait" per 100 words, for steering in the Outside Truth (FIG 1) and Reasoned Truth (FIG 2) directions.
