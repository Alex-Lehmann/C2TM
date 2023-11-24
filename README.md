# C2TM
A Python package implementing a contrastive contextualized topic model (C2TM) for cross-lingual topic modeling, including zero-shot learning transfer to unseen languages.

# Basic Usage
1. Get the code with `git clone https://github.com/Alex-Lehmann/C2TM.git`.
2. Load the main class with `from c2tm.core import C2TM`.
3. Instantiate a C2TM with `model = C2TM(<number of topics>, <first language>, <second language>)`.
4. Load a parallel training corpora with `model.ingest_corpus(<documents in first language>, <documents in second language>)`.
5. Fit the model with `model.fit(<number of training epochs>)`.
6. Examine the top words in each topic with `model.get_topic_words(<language for results>)`.

# Author
[Alex Lehmann](mailto:alexlehmann@cmail.carleton.ca), The DANG Lab, Carleton University.

# Version History
* 0.0.1
  * Initial release.

# License
This project is licensed under the GNU General Public License v3.0 - see the LICENSE.md file for more details.

# Achknowledgements
This project is based on the excellent prior work found in:
* Federico Bianchi, Silvia Terragni, Dirk Hovy, Debora Nozza, and Elisabetta Fersini. 2021. [Cross-lingual Contextualized Topic Models with Zero-shot Learning](https://aclanthology.org/2021.eacl-main.143). In _Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume_, pages 1676-1683, Online. Association for Computational Linguistics.
* Yu Wang,. Hengrui Zhang, Zhiwei Liu, Liangwei Yang, Philip S. Yu. 2022. [ContrastVAE: Contrastive Variational AutoEncoder for Sequential Recommendation](https://dl.acm.org/doi/abs/10.1145/3511808.3557268). In _Proceedings of the 31st ACM International Conference on Information & Knowledge Management_, pages 2056-2066, Atlanta, GA, USA. Association for Computating Machinery.
