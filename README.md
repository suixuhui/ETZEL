# ETZEL

## Data Preparation
* Cleate a `data` fold
* Prepare the zero-shot entity linking data following <https://github.com/facebookresearch/BLINK>, place it under `data`
* Prepare the ultra-fine entity typing data from <https://www.cs.utexas.edu/~eunsol/html_pages/open_entity.html>, place it under `data`

## Run
1. Encoding type and its description: `python typing/encode_types.py`
2. Train the ultra-fine entity typing model: `python typing/main.py`
3. Generate type information of the zero-shot entity linking data: `python typing/main.py --mode generate --load_modal "./models/berttype"`
4. Merge all type information: `python generate_types.py`
5. Run zero-shot entity linking candatate generation: 


If you use our code in your work, please cite us.

*Xuhui Sui, Ying Zhang, Kehui Song, Baohang Zhou, Guoqing Zhao, Xin Wei and Xiaojie Yuan. 2022. Improving Zero-Shot Entity Linking Candidate Generation with Ultra-Fine Entity Type Information. 2022 International Conference on Computational Linguistics (COLING 2022).*
