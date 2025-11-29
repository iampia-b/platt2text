import argparse
import numpy
import seaborn

from matplotlib import pyplot
from pathlib import Path
from transformers import WhisperForConditionalGeneration, WhisperTokenizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_language_embeddings(model: WhisperForConditionalGeneration, tokenizer: WhisperTokenizer,lang_codes: list):

    embeddings = model.get_input_embeddings()
    
    lang_embeddings = {}
    token_ids = {}
    
    for lang in lang_codes:
        token = f"<|{lang}|>"
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            print(f"Language code {lang} not found in tokenizer vocabulary.")
            continue
        embedding_vector = embeddings.weight[token_id].detach().cpu().numpy()
        lang_embeddings[lang] = embedding_vector
        token_ids[lang] = token_id
    
    return lang_embeddings, token_ids

def plot_cosine_similarities(similarity_matrix, lang_list, output_path):
    pyplot.figure(figsize=(10, 8))
    ax = seaborn.heatmap(
        similarity_matrix,
        xticklabels=lang_list,
        yticklabels=lang_list,
        annot=True,
        cmap='viridis',
        fmt=".4f",
        annot_kws={"size": 15}
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)
    pyplot.xticks(fontsize=15)
    pyplot.yticks(fontsize=15)
    pyplot.savefig(output_path, dpi=300, bbox_inches='tight')
    pyplot.close()

def plot_similarity_to_default(default_lang, similarity_matrix, lang_list, output_path):
    idx = lang_list.index(default_lang)
    sims = similarity_matrix[idx, :]

    other_langs = []
    other_sims = []
    for lang, val in zip(lang_list, sims):
        if lang != default_lang:
            other_langs.append(lang)
            other_sims.append(val)

    other_sims = numpy.array(other_sims)
    sort_idx = numpy.argsort(-other_sims)
    sorted_langs = [other_langs[i] for i in sort_idx]
    sorted_sims = other_sims[sort_idx]

    num_langs = len(sorted_langs)
    fig_width = max(6, 2 + num_langs * 0.5)

    pyplot.figure(figsize=(fig_width, 8))
    x = numpy.arange(len(sorted_langs))
    colors = ['red' if 'nds' in lang.lower() else 'blue' for lang in sorted_langs]

    pyplot.scatter(x, sorted_sims, c=colors)
    pyplot.xticks(x, sorted_langs, fontsize=15)
    pyplot.yticks(fontsize=15)

    pyplot.tight_layout()
    pyplot.savefig(output_path, dpi=300, bbox_inches='tight')
    pyplot.close()


def main():
    parser = argparse.ArgumentParser()
    # model paths
    parser.add_argument("--only_pretrained", type=bool, default=False) # if true, only use the pretrained model embeddings for all languages
    parser.add_argument("--model_a_dir", type=str)
    parser.add_argument("--model_a_name", type=str)
    parser.add_argument("--model_b_dir", type=str)
    parser.add_argument("--model_b_name", type=str)
    parser.add_argument("--default_code", type=str)
    parser.add_argument("--lang_codes", nargs='+', required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    args = parser.parse_args()

    # load models and tokenizers
    print(f"Loading default model (openai/whisper-large-v3) from openai/whisper-large-v3")
    model_0 = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
    tokenizer_0 = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")

    if not args.only_pretrained:
        # load models and tokenizers
        print(f"Loading model A ({args.model_a_name}) from {args.model_a_dir}")
        model_a = WhisperForConditionalGeneration.from_pretrained(args.model_a_dir)
        tokenizer_a = WhisperTokenizer.from_pretrained(args.model_a_dir)

        print(f"Loading model B ({args.model_b_name}) from {args.model_b_dir}")
        model_b = WhisperForConditionalGeneration.from_pretrained(args.model_b_dir)
        tokenizer_b = WhisperTokenizer.from_pretrained(args.model_b_dir)

    # combine embeddings
    all_embeddings = {}
    all_token_ids = {}
    
    # add default code to lang_codes 
    all_lang_codes = [args.default_code]
    for lang in args.lang_codes:
        if lang not in all_lang_codes:
            all_lang_codes.append(lang)

    # extracting embeddngs
    embeddings_0, token_ids_0 = extract_language_embeddings(model_0, tokenizer_0, all_lang_codes)
    for lang, emb in embeddings_0.items():
        if lang != "nds":
            all_embeddings[f"{lang}"] = emb
            all_token_ids[f"{lang}"] = token_ids_0[lang]

    if not args.only_pretrained:    
        embeddings_a, token_ids_a = extract_language_embeddings(model_a, tokenizer_a, all_lang_codes)
        for lang, emb in embeddings_a.items():
            if lang == "nds":
                all_embeddings[f"{lang} ({args.model_a_name})"] = emb
                all_token_ids[f"{lang} ({args.model_a_name})"] = token_ids_a[lang]

        embeddings_b, token_ids_b = extract_language_embeddings(model_b, tokenizer_b, all_lang_codes)
        for lang, emb in embeddings_b.items():
            if lang == "nds":
                all_embeddings[f"{lang} ({args.model_b_name})"] = emb
                all_token_ids[f"{lang} ({args.model_b_name})"] = token_ids_b[lang]

    # calculate pairwise cosine similarities
    lang_list = list(all_embeddings.keys())
    embedding_matrix = numpy.array([all_embeddings[lang] for lang in lang_list])
    similarity_matrix = cosine_similarity(embedding_matrix)

    print("\nCosine Similarity Matrix:")
    print("Embeddings:", lang_list)
    print(similarity_matrix)

    # save similarity matrix
    args.output_dir.mkdir(parents=True, exist_ok=True)
    numpy.savetxt(args.output_dir / "language_embedding_similarities.csv", similarity_matrix, delimiter=",", fmt="%.4f", header=",".join(lang_list), comments='')

    # create plot of cosine similarities
    plot_cosine_similarities(similarity_matrix, lang_list, args.output_dir / "language_embedding_similarities.png")
    
    # plot similarities to default language
    default_key = f"{args.default_code}"
    if default_key in lang_list:
        plot_similarity_to_default(default_key, similarity_matrix, lang_list, args.output_dir / f"similarity_to_{default_key}.png") 


if __name__ == "__main__":
    main()