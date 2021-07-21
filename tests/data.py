import logging
from tqdm import tqdm
from transformers import AutoTokenizer

from t5qg.data import get_dataset
from t5qg.sentence_split import SentSplit

# print(SentSplit().splitter("""
# "Nintendo Co., Ltd. is a Japanese multinational consumer electronics and video game company headquartered in Kyoto. The company was founded in 1889 as Nintendo Karuta by craftsman Fusajiro Yamauchi and originally produced handmade hanafuda playing cards. After venturing into various lines of business during the 1960s and acquiring a legal status as a public company under the current company name, Nintendo distributed its first video game console, the Color TV-Game, in 1977. It gained international recognition with the release of Donkey Kong in 1981 and the Nintendo Entertainment System and Super Mario Bros. in 1985. Since then, Nintendo has produced some of the most successful consoles in the video game industry, such as the Game Boy, the Super Nintendo Entertainment System, the Nintendo DS, the Wii, and the Nintendo Switch. It has created numerous major franchises, including Mario, Donkey Kong, The Legend of Zelda, Pokémon, Kirby, Metroid, Fire Emblem, Animal Crossing, Splatoon, Star Fox, Xenoblade Chronicles, and Super Smash Bros. The character of Mario is internationally recognisable, and serves as the company's mascot.
# """))

level = logging.DEBUG
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=level, datefmt='%Y-%m-%d %H:%M:%S')

# get_dataset('tydiqa', language=['en'], split='dev', cache_dir='./cache')
tokenizer = AutoTokenizer.from_pretrained('google/mt5-small')
for data_name in ['squad']:
    print(data_name)
    get_dataset(data_name, split='test')
    get_dataset(data_name, split='dev')
    get_dataset(data_name, split='train')
    input()
    in_text, out_text = get_dataset(data_name, split='dev', task_type='ans_ext', no_prefix=False, cache_dir='cache')
    total_in, total_out = [], []
    for i in tqdm(in_text):
        input(i)

        # n_in = len(tokenizer.encode(i['source_text']))
        # n_out = len(tokenizer.encode(i['target_text']))
        # total_in.append(n_in)
        # total_out.append(n_out)

    print('\t input token : {} (<{})'.format(sum(total_in)/len(total_in), max(total_in)))
    print('\t output token: {} (<{})'.format(sum(total_out)/len(total_out), max(total_out)))
