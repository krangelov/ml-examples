import pgf
import torch
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name = "meta-llama/Llama-3.2-3B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             torch_dtype=torch.float16)
model = model.to("cuda")

gr = pgf.readPGF("../GF/gf-wordnet/build/ParseEng.pgf")
langs = gr.languages

def groupby(xs):
    groups = {}
    for x in xs:
        lin = langs["ParseEng"].linearize(pgf.ExprFun(x))
        groups.setdefault(lin,[]).append(x)
    return groups

adjs  = groupby([adj  for adj  in gr.functionsByCat("A") if len(gr.functionType(adj).hypos) == 0])
nouns = groupby([noun for noun in gr.functionsByCat("N") if len(gr.functionType(noun).hypos) == 0])

prob_threshold = 24
batch_size = 1000

with open("output.txt","w") as f:
    for adj_str,adj_lst in adjs.items():
        adj_ids = tokenizer(adj_str, return_tensors="pt").to(model.device)

        with torch.no_grad():
            adj_out = model(**adj_ids, use_cache=True)
        adj_log_probs = torch.log_softmax(adj_out.logits,dim=-1)

        token_log_probs = adj_log_probs.gather(
            2, adj_ids["input_ids"][:,1:].unsqueeze(-1)
        ).squeeze(-1)
        adj_log_prob = -token_log_probs.sum().item()
        del token_log_probs

        adj_cache = adj_out.past_key_values

        sample = []
        for noun_str,noun_lst in nouns.items():
            noun_ids = tokenizer(" "+noun_str, return_tensors="pt",add_special_tokens=False).to(model.device)
            noun_ids = noun_ids["input_ids"].squeeze(0)

            noun_log_prob = -adj_log_probs[0,-1,noun_ids[0]].item()
            sentence_log_prob = adj_log_prob + noun_log_prob
            if sentence_log_prob < prob_threshold:
                sample.append((adj_lst,noun_lst,noun_ids,sentence_log_prob))

        if not sample:
            continue

        print(len(sample))
        for i in range(0,len(sample),batch_size):
            noun_ids = [noun_ids for _,_,noun_ids,_ in sample[i:i+batch_size]]
            noun_ids = torch.nn.utils.rnn.pad_sequence(noun_ids,
                                                       batch_first=True,
                                                       padding_value=tokenizer.eos_token_id)
            batch_cache = copy.deepcopy(adj_cache)
            batch_cache.batch_repeat_interleave(noun_ids.shape[0])
            with torch.no_grad():
                noun_out = model(input_ids=noun_ids,
                                 past_key_values=batch_cache)
            noun_log_probs = torch.log_softmax(noun_out.logits,dim=-1)
            del noun_out

            for j in range(noun_ids.shape[0]):
                adj_lst,noun_lst,ids,sentence_log_prob = sample[i+j]

                token_log_probs = noun_log_probs[j].gather(
                    1, ids[1:].unsqueeze(-1)
                ).squeeze(-1)
                sentence_log_prob += -token_log_probs.sum().item()

                print(adj_lst,noun_lst,sentence_log_prob)
                sample[i+j] = (adj_lst,noun_lst,sentence_log_prob)

            del noun_log_probs

        sample.sort(key=lambda x: x[2])
        for adj_lst,noun_lst,sentence_log_prob in sample:
            if sentence_log_prob < prob_threshold:
                for adj in adj_lst:
                    for noun in noun_lst:
                        e = pgf.readExpr(f"AdjCN (PositA {adj}) (UseN {noun})")
                        f.write(str(e)+"\t"+str(sentence_log_prob)+"\n")

        f.flush()
