import os
from t5qg import T5

MODEL = os.getenv('MODEL', 'asahi417/question-generation-squad-t5-small')
MAX_LENGTH = int(os.getenv('MAX_LENGTH', 512))
MAX_LENGTH_OUTPUT = int(os.getenv('MAX_LENGTH_OUTPUT', 32))

qg_model = T5(MODEL, MAX_LENGTH, MAX_LENGTH_OUTPUT)
sample = "Nintendo Co., Ltd. is a Japanese multinational consumer electronics and video game company headquartered in Kyoto. " \
         "The company was founded in 1889 as Nintendo Karuta by craftsman Fusajiro Yamauchi and originally produced handmade " \
         "hanafuda playing cards. After venturing into various lines of business during the 1960s and acquiring a legal status " \
         "as a public company under the current company name, Nintendo distributed its first video game console, the Color TV-Game," \
         " in 1977. It gained international recognition with the release of Donkey Kong in 1981 and the Nintendo Entertainment " \
         "System and Super Mario Bros. in 1985. Since then, Nintendo has produced some of the most successful consoles in the " \
         "video game industry, such as the Game Boy, the Super Nintendo Entertainment System, the Nintendo DS, the Wii, and " \
         "the Nintendo Switch. It has created numerous major franchises, including Mario, Donkey Kong, The Legend of Zelda, " \
         "Pokémon, Kirby, Metroid, Fire Emblem, Animal Crossing, Splatoon, Star Fox, Xenoblade Chronicles, and Super Smash " \
         "Bros. The character of Mario is internationally recognisable, and serves as the company's mascot."
squad_qg = "generate question: Architecturally, the school has a Catholic character. Atop the Main Building's gold dome is a golden " \
           "statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with" \
           " arms upraised with the legend Venite Ad Me Omnes. Next to the Main Building is the Basilica of the Sacred Heart. " \
           "Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the " \
           "grotto at Lourdes, France where the Virgin Mary reputedly appeared to  " \
           "<hl> Saint Bernadette Soubirous <hl>  in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary."

print(qg_model.generate_qa(sample))
print(qg_model.generate_q([squad_qg]*3))

a = qg_model.generate_a(sample)
print(a)

q = qg_model.generate_q([sample] * len(a), list_answer=a)
print(q)



