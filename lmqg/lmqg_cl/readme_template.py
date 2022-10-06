"""
TODO: Update paper & citation info
"""
import os
import json
import re
from os.path import join as pj
from typing import Dict
from glob import glob
from lmqg.language_model import TASK_PREFIX
from datasets import load_dataset

bib = """
TBA
"""

version_description = {
    'default': "This model is fine-tuned without parameter search (default configuration is taken from [ERNIE-GEN](https://arxiv.org/abs/2001.11314)).",
    'no-answer': "This model is fine-tuned without answer information, i.e. generate a question only given a paragraph (note that normal model is fine-tuned to generate a question given a pargraph and an associated answer in the paragraph).",
    'no-paragraph': "This model is fine-tuned without pargraph information but only the sentence that contains the answer.",
    'multitask': "This model is fine-tuned on the answer extraction task as well as the question generation."
}

sample_qg_dict = {
    "en": [
        "<hl> Beyonce <hl> further expanded her acting career, starring as blues singer Etta James in the 2008 musical biopic, Cadillac Records.",
        "Beyonce further expanded her acting career, starring as blues singer <hl> Etta James <hl> in the 2008 musical biopic, Cadillac Records.",
        "Beyonce further expanded her acting career, starring as blues singer Etta James in the 2008 musical biopic,  <hl> Cadillac Records <hl> ."
        ],
    "ja": [
        "ゾフィーは貴族出身ではあったが王族出身ではなく、ハプスブルク家の皇位継承者であるフランツ・フェルディナントとの結婚は貴賤結婚となった。皇帝フランツ・ヨーゼフは、2人の間に生まれた子孫が皇位を継がないことを条件として結婚を承認していた。視察が予定されている<hl>6月28日<hl>は2人の14回目の結婚記念日であった。",
        "『クマのプーさん』の物語はまず1925年12月24日、『イヴニング・ニュース』紙のクリスマス特集号に短編作品として掲載された。これは『クマのプーさん』の第一章にあたる作品で、このときだけは挿絵をJ.H.ダウドがつけている。その後作品10話と挿絵が整い、刊行に先駆けて「イーヨーの誕生日」のエピソードが1926年8月に『ロイヤルマガジン』に、同年10月9日に『ニューヨーク・イヴニング・ポスト』紙に掲載されたあと、同年10月14日にロンドンで(メシュエン社)、21日にニューヨークで(ダットン社)『クマのプーさん』が刊行された。前著『ぼくたちがとてもちいさかったころ』がすでに大きな成功を収めていたこともあり、イギリスでは初版は前著の7倍に当たる<hl>3万5000部<hl>が刷られた。他方のアメリカでもその年の終わりまでに15万部を売り上げている。ただし依然として人気のあった前著を売り上げで追い越すには数年の時間を要した。",
        "フェルメールの作品では、17世紀のオランダの画家、ヨハネス・フェルメールの作品について記述する。フェルメールの作品は、疑問作も含め<hl>30数点<hl>しか現存しない。現存作品はすべて油彩画で、版画、下絵、素描などは残っていない。以下には若干の疑問作も含め、37点の基本情報を記載し、各作品について略説する。収録順序、推定制作年代は『「フェルメールとその時代展」図録』による。日本語の作品タイトルについては、上掲図録のほか、『「フェルメール展」図録』、『フェルメール生涯と作品』による。便宜上「1650年代の作品」「1660年代の作品」「1670年代の作品」の3つの節を設けたが、フェルメールの作品には制作年代不明のものが多く、推定制作年代については研究者や文献によって若干の差がある。",
    ],
    "ru": [
        "Нелишним будет отметить, что, развивая это направление, Д. И. Менделеев, поначалу априорно выдвинув идею о температуре, при которой высота мениска будет нулевой, <hl> в мае 1860 года <hl> провёл серию опытов.",
        "Однако, франкоязычный <hl> Квебек <hl> практически никогда не включается в состав Латинской Америки.",
        "Классическим примером международного синдиката XX века была группа компаний <hl> Де Бирс <hl> , которая в 1980-е годы контролировала до 90 % мировой торговли алмазами."
    ],
    "ko": [
        "1990년 영화 《 <hl> 남부군 <hl> 》에서 단역으로 영화배우 첫 데뷔에 이어 같은 해 KBS 드라마 《지구인》에서 단역으로 출연하였고 이듬해 MBC 《여명의 눈동자》를 통해 단역으로 출연하였다.",
        "백신이 없기때문에 예방책은 <hl> 살충제 <hl> 를 사용하면서 서식 장소(찻찬 받침, 배수로, 고인 물의 열린 저장소, 버려진 타이어 등)의 수를 줄임으로써 매개체를 통제할 수 있다.",
        "<hl> 원테이크 촬영 <hl> 이기 때문에 한 사람이 실수를 하면 처음부터 다시 찍어야 하는 상황이 발생한다."
    ],
    "es": [
        "del <hl> Ministerio de Desarrollo Urbano <hl> , Gobierno de la India.",
        "a <hl> noviembre <hl> , que es también la estación lluviosa.",
        "como <hl> el gobierno de Abbott <hl> que asumió el cargo el 18 de septiembre de 2013."
    ],
    "fr": [
        "Créateur » (Maker), lui aussi au singulier, « <hl> le Suprême Berger <hl> » (The Great Shepherd) ; de l'autre, des réminiscences de la théologie de l'Antiquité : le tonnerre, voix de Jupiter, « Et souvent ta voix gronde en un tonnerre terrifiant », etc.",
        "Ce black dog peut être lié à des évènements traumatisants issus du monde extérieur, tels que son renvoi de l'Amirauté après la catastrophe des Dardanelles, lors de la <hl> Grande Guerre <hl> de 14-18, ou son rejet par l'électorat en juillet 1945.",
        "contre <hl> Normie Smith <hl> et 15 000 dollars le 28 novembre 1938."
    ],
    "de": [
        "Empfangs- und Sendeantenne sollen in ihrer Polarisation übereinstimmen, andernfalls <hl> wird die Signalübertragung stark gedämpft. <hl>",
        "das erste weltweit errichtete Hermann Brehmer <hl> 1855 <hl> im niederschlesischen ''Görbersdorf'' (heute Sokołowsko, Polen).",
        "Er muss Zyperngrieche sein und wird direkt für <hl> fünf Jahre <hl> gewählt (Art. 43 Abs. 1 der Verfassung) und verfügt über weitreichende Exekutivkompetenzen."
    ],
    "it": [
        "<hl> Dopo il 1971 <hl> , l' OPEC ha tardato ad adeguare i prezzi per riflettere tale deprezzamento.",
        "L' individuazione del petrolio e lo sviluppo di nuovi giacimenti richiedeva in genere <hl> da cinque a dieci anni <hl> prima di una produzione significativa.",
        "il <hl> Giappone <hl> è stato il paese più dipendente dal petrolio arabo."
    ]
}

sample_qa_dict = {
    "en": [
        "<hl> Beyonce further expanded her acting career, starring as blues singer Etta James in the 2008 musical biopic, Cadillac Records. <hl> Her performance in the film received praise from critics, and she garnered several nominations for her portrayal of James, including a Satellite Award nomination for Best Supporting Actress, and a NAACP Image Award nomination for Outstanding Supporting Actress.",
        "Beyonce further expanded her acting career, starring as blues singer Etta James in the 2008 musical biopic, Cadillac Records. <hl> Her performance in the film received praise from critics, and she garnered several nominations for her portrayal of James, including a Satellite Award nomination for Best Supporting Actress, and a NAACP Image Award nomination for Outstanding Supporting Actress. <hl>"
    ],
    "ja": [
        "『クマのプーさん』の物語はまず1925年12月24日、『イヴニング・ニュース』紙のクリスマス特集号に短編作品として掲載された。これは『クマのプーさん』の第一章にあたる作品で、このときだけは挿絵をJ.H.ダウドがつけている。その後作品10話と挿絵が整い、刊行に先駆けて「イーヨーの誕生日」のエピソードが1926年8月に『ロイヤルマガジン』に、同年10月9日に『ニューヨーク・イヴニング・ポスト』紙に掲載されたあと、同年10月14日にロンドンで(メシュエン社)、21日にニューヨークで(ダットン社)『クマのプーさん』が刊行された。<hl>前著『ぼくたちがとてもちいさかったころ』がすでに大きな成功を収めていたこともあり、イギリスでは初版は前著の7倍に当たる3万5000部が刷られた。<hl>他方のアメリカでもその年の終わりまでに15万部を売り上げている。ただし依然として人気のあった前著を売り上げで追い越すには数年の時間を要した。",
        "フェルメールの作品では、17世紀のオランダの画家、ヨハネス・フェルメールの作品について記述する。フェルメールの作品は、疑問作も含め30数点しか現存しない。<hl>現存作品はすべて油彩画で、版画、下絵、素描などは残っていない。以下には若干の疑問作も含め、37点の基本情報を記載し、各作品について略説する。<hl>収録順序、推定制作年代は『「フェルメールとその時代展」図録』による。日本語の作品タイトルについては、上掲図録のほか、『「フェルメール展」図録』、『フェルメール生涯と作品』による。便宜上「1650年代の作品」「1660年代の作品」「1670年代の作品」の3つの節を設けたが、フェルメールの作品には制作年代不明のものが多く、推定制作年代については研究者や文献によって若干の差がある。"
    ],
    "ru": [
        "<hl> в английском языке в нарицательном смысле применяется термин rapid transit (скоростной городской транспорт), однако употребляется он только тогда, когда по смыслу невозможно ограничиться названием одной конкретной системы метрополитена. <hl> в остальных случаях используются индивидуальные названия: в лондоне — london underground, в нью-йорке — new york subway, в ливерпуле — merseyrail, в вашингтоне — washington metrorail, в сан-франциско — bart и т. п. в некоторых городах применяется название метро (англ. metro) для систем, по своему характеру близких к метро, или для всего городского транспорта (собственно метро и наземный пассажирский транспорт (в том числе автобусы и трамваи)) в совокупности.",
        "Вопреки ожиданиям, объединение денежных систем республик не привело к уменьшению инфляции. Напротив, закдензнаки стали невероятно быстро обесцениваться, особенно в 1924 году. Для обеспечения денежного рынка приходилось увеличивать эмиссию закдензнаков и выпускать в оборот купюры невероятно больших номиналов. <hl> Так, в период с 1 января по 20 марта 1924 года были введены в оборот купюры достоинством 25 000 000 рублей, затем — 250 000 000 рублей. <hl> И, наконец, в апреле 1924 года были выпущены купюры миллиардного достоинства (в просторечии лимард)."
    ],
    "ko": [
        "또한 스피어스는 많은 새로운 여성 아티스트들에게 영향을 끼쳤는데, 대표적으로 데미 로바토, 케이티 페리, 크리스티니아 드바지, 레이디 가가, 리틀 부츠, 셀레나 고메즈 & 더씬, 픽시 로트 이 있다. 2007년 비욘세 놀스는 Total Request Live와의 인터뷰에서 '나는 브리트니를 사랑하고 팬이에요. 특히 새 앨범 Blackout을 좋아해요'라고 말했다. 린제이 로한은 '언제나 브리트니 스피어스에게 영감을 받는다. 학창시절 그녀처럼 타블로이드에 오르기를 꿈꿔왔다'고 말하며 롤 모델로 꼽았다. 스피어스는 현대 음악가들에게 음악적 영감으로 언급되기도 했다. <hl> 마일리 사이러스는 자신의 히트곡 Party in the U.S.A. 가 브리트니에게 영감과 영향을 받은 곡이라고 밝혔다. <hl> 베리 매닐로우의 앨범 15 Minutes 역시 브리트니에게 영감을 얻었다고 언급되었다.",
        "지난 22일 아프리카TV는 BJ 철구가 서비스 정지 처분을 받았음을 밝혔다. 서비스 정지 처분을 사유는 철구가 10대 청소년에게 유해한 장면을 방송으로 내보냈기 때문이었다. 문제가 된 장면은 BJ 철구가 미성년자는 시청할 수 없게 하는 19세 시청 가능 설정을 하지 않은 채 흡연하는 모습을 여과 없이 드러낸 장면이다. 아프리카TV는 청소년 보호 정책의 '청소년들이 해로운 환경으로부터 보호받을 수 있도록 조치한다'라고 조항을 근거로 철구에게 서비스 정지 처분을 내렸다. 흡연 이외에 음주 방송 등도 19세 시청 가능 설정을 해야만 방송할 수 있다. <hl> 게다가 철구의 방송 정지 처분은 이번에 처음이 아니라 16번 째기 때문에 더욱더 논란이 되고 있다. <hl>"
    ],
    "es": [
        "<hl> En la diáspora somalí, múltiples eventos islámicos de recaudación de fondos se llevan a cabo cada año en ciudades como Birmingham, Londres, Toronto y Minneapolis, donde los académicos y profesionales somalíes dan conferencias y responden preguntas de la audiencia. <hl> El propósito de estos eventos es recaudar dinero para nuevas escuelas o universidades en Somalia, para ayudar a los somalíes que han sufrido como consecuencia de inundaciones y / o sequías, o para reunir fondos para la creación de nuevas mezquitas como.",
        "<hl> Los estudiosos y los histori a dores están divididos en cuanto a qué evento señala el final de la era helenística. <hl> El período helenístico se puede ver que termina con la conquista final del corazón griego por Roma en 146 a. C. tras la guerra aquea, con la derrota final del reino ptolemaico en la batalla de Actium en 31 a. Helenístico se distingue de helénico en que el primero abarca toda la esfera de influencia griega antigua directa, mientras que el segundo se refiere a la propia Grecia."
    ],
    "fr": [
        "Pourtant, la strophe spensérienne, utilisée cinq fois avant que ne commence le chœur, constitue en soi un vecteur dont les répétitions structurelles, selon Ricks, relèvent du pur lyrisme tout en constituant une menace potentielle. Après les huit sages pentamètres iambiques, l'alexandrin final <hl> permet une pause <hl>, « véritable illusion d'optique » qu'accentuent les nombreuses expressions archaïsantes telles que did swoon, did seem, did go, did receive, did make, qui doublent le prétérit en un temps composé et paraissent à la fois « très précautionneuses et très peu pressées ».",
        "Néanmoins, une fois encore, l'arithmétique modulaire est insuffisante pour venir à bout du théorème. Dirichlet utilise de nombreuses techniques analytiques, comme les séries entières et l'analyse complexe. Le fruit de ces travaux donne naissance à une nouvelle branche des mathématiques : la théorie analytique des nombres. L'un des points cruciaux de cette théorie provient de l'unique article de <hl> Bernhard Riemann <hl> en théorie des nombres : Sur le nombre de nombres premiers inférieurs à une taille donnée. Il conjecture une localisation des racines de sa fonction ζ. La recherche de la position des racines, initiée par Dirichlet, devient une préoccupation centrale et reste l'une des conjectures pressenties comme les plus difficiles des mathématiques de notre époque.",
    ],
    "de": [
        "Sommerzeit <hl> Frühling <hl>: Umstellung von Normalzeit auf Sommerzeit – die Uhr wird um eine Stunde ''vor''gestellt. Herbst: Umstellung von Sommerzeit auf Normalzeit – die Uhr wird um eine Stunde ''zurück''gestellt. Als Sommerzeit wird die gegenüber der Zonenzeit meist um eine Stunde vorgestellte Uhrzeit bezeichnet, die während eines bestimmten Zeitraums im Sommerhalbjahr (und oft auch etwas darüber hinaus) als gesetzliche Zeit dient. Eine solche Regelung wird fast nur in Ländern der gemäßigten Zonen angewandt. Die mitteleuropäische Sommerzeit beginnt am letzten Sonntag im März um 2:00 Uhr MEZ, indem die Stundenzählung um eine Stunde von 2:00 Uhr auf 3:00 Uhr vorgestellt wird. Sie endet jeweils am letzten Sonntag im Oktober um 3:00 Uhr MESZ, indem die Stundenzählung um eine Stunde von 3:00 Uhr auf 2:00 Uhr zurückgestellt wird.",
        "Iran === Landwirtschaft === Die landwirtschaftliche Nutzfläche beträgt trotz zahlreicher Gebirge und Wüsten 10 % der Landesfläche, wobei ein Drittel künstlich bewässert wird. Die Landwirtschaft ist einer der größten Arbeitgeber des Landes. Wichtige Produkte sind Pistazien, Weizen, Reis, Zucker, Baumwolle, Früchte, Nüsse, Datteln, Wolle und Kaviar. Seit der Revolution von 1979 wurde der Anbau von Weintrauben wegen des islamischen Alkoholverbots auf den 200.000 Hektar Rebfläche fast vollständig auf Tafeltrauben und Rosinen umgestellt. Bei Rosinen ist <hl> der Iran <hl> inzwischen nach der Türkei der zweitgrößte Exporteur der Welt, bei Safran mit ungefähr 90 % Marktanteil des globalen Bedarfs mit Abstand der größte."
    ],
    "it": [
        "<hl> Il 6 ottobre 1973 , la Siria e l' Egitto, con il sostegno di altre nazioni arabe, lanciarono un attacco a sorpresa su Israele, su Yom Kippur. <hl> Questo rinnovo delle ostilità nel conflitto arabo-israeliano ha liberato la pressione economica sottostante sui prezzi del petrolio. All' epoca, l' Iran era il secondo esportatore mondiale di petrolio e un vicino alleato degli Stati Uniti. Settimane più tardi, lo scià d' Iran ha detto in un' intervista: Naturalmente[il prezzo del petrolio] sta andando a salire Certamente! E come! Avete[Paesi occidentali] aumentato il prezzo del grano che ci vendete del 300 per cento, e lo stesso per zucchero e cemento.",
        "<hl> Furono introdotti autocarri compatti, come la Toyota Hilux e il Datsun Truck, seguiti dal camion Mazda (venduto come il Ford Courier), e l' Isuzu costruito Chevrolet LUV. <hl> Mitsubishi rebranded il suo Forte come Dodge D-50 pochi anni dopo la crisi petrolifera. Mazda, Mitsubishi e Isuzu avevano partnership congiunte rispettivamente con Ford, Chrysler e GM. In seguito i produttori americani introdussero le loro sostituzioni nazionali (Ford Ranger, Dodge Dakota e la Chevrolet S10/GMC S-15), ponendo fine alla loro politica di importazione vincolata."
    ]
}

language_dict = {
    "qg_squad": 'en',
    "qg_frquad": 'fr',
    "qg_itquad": 'it',
    "qg_koquad": 'ko',
    "qg_jaquad": 'ja',
    "qg_ruquad": 'ru',
    "qg_esquad": 'es',
}


def format_metric(dataset, dataset_type, metric):
    return f"""  - task:
      name: Text2text Generation
      type: text2text-generation
    dataset:
      name: {dataset}
      type: {dataset_type}
      args: {dataset_type}
    metrics:
    - name: BLEU4
      type: bleu4
      value: {metric["test"]["Bleu_4"]}
    - name: ROUGE-L
      type: rouge-l
      value: {metric["test"]["ROUGE_L"]}
    - name: METEOR
      type: meteor
      value: {metric["test"]["METEOR"]}
    - name: BERTScore
      type: bertscore
      value: {metric["test"]["BERTScore"]}
    - name: MoverScore
      type: moverscore
      value: {metric["test"]["MoverScore"]}"""


def format_usage(model_name, sample_qg, sample_qa):
    qg_usage = f"""
from transformers import pipeline

model_path = '{model_name}'
pipe = pipeline("text2text-generation", model_path)

# Question Generation
question = pipe('{sample_qg[0]}')"""
    if len(sample_qa) > 0:
        qg_usage += f"""
# Answer Extraction
answer = pipe('{sample_qa[0]}')"""
    return qg_usage


def get_readme(model_name: str, model_checkpoint):
    with open(pj(model_checkpoint, "trainer_config.json")) as f:
        config = json.load(f)
    config_text = "\n".join([f" - {k}: {v}" for k, v in config.items()])
    language_model = config['model']
    dataset = config['dataset_path']
    prefix_types = config['prefix_types']
    dataset_name = config['dataset_name']
    dataset_alias = os.path.basename(dataset)
    la = language_dict[dataset_alias] if dataset_alias in language_dict else 'en'

    # model_version
    eval_file = "metric.first.sentence.paragraph_answer.question"
    add_info = []
    _sample_qg = sample_qg_dict[la]
    if model_name.endswith('no-answer'):
        _sample_qg = ["<hl> " + re.sub(r'\s+', ' ', i.replace('<hl>', '')) + " <hl>" for i in _sample_qg]
        add_info.append(version_description['no-answer'])
        eval_file = "metric.first.sentence.paragraph_sentence.question"
    elif model_name.endswith('no-paragraph'):
        add_info.append(version_description['no-paragraph'])
        eval_file = "metric.first.sentence.sentence_answer.question"
    elif model_name.endswith('default'):
        add_info.append(version_description['default'])
    if dataset_alias in ['qg_subjqa', 'qg_squadshifts'] and 'vanilla' not in model_name:
        add_info.append(f"This model is continuously fine-tuned with [{language_model}](https://huggingface.co/{language_model}).")
    if model_name.endswith('multitask'):
        add_info.append(version_description['multitask'])
    add_info = ' '.join(add_info)

    # get widget
    sample_qa = []
    if prefix_types is not None and len(prefix_types) > 1:  # multitask
        tags = "- question generation\n- answer extraction"
        sample_qg = [f'{TASK_PREFIX["qg"]}: {i}' for i in _sample_qg]
        sample_qa = [f'{TASK_PREFIX["ae"]}: {i}' for i in sample_qa_dict[la]]
        widget = '\n'.join([f"""- text: "{i}"\n  example_title: "Question Generation Example {n + 1}" """ for n, i in enumerate(sample_qg)])
        widget += '\n' + '\n'.join([f"""- text: "{i}"\n  example_title: "Answer Extraction Example {n + 1}" """ for n, i in enumerate(sample_qa_dict[la])])
    elif prefix_types is not None:
        tags = "- question generation"
        sample_qg = [f'{TASK_PREFIX["qg"]}: {i}' for i in _sample_qg]
        widget = '\n'.join([f"""- text: "{i}"\n  example_title: "Question Generation Example {n + 1}" """ for n, i in enumerate(sample_qg)])
    else:
        tags = "- question generation"
        sample_qg = _sample_qg
        widget = '\n'.join([f"""- text: "{i}"\n  example_title: "Question Generation Example {n + 1}" """ for n, i in enumerate(sample_qg)])

    # usage
    usage = format_usage(model_name, sample_qg, sample_qa)

    # metric
    with open(pj(model_checkpoint, "eval", f"{eval_file}.{dataset.replace('/', '_')}.{dataset_name}.json")) as f:
        metric = json.load(f)
        metric_main = [dataset, dataset_name, metric]
    # metric for ood
    metrics_ood = []
    for i in glob(pj(model_checkpoint, "eval_ood", f"{eval_file}.*.json")):
        ood_data = os.path.basename(i).split(f"{eval_file}.")[-1].replace('.json', '')
        _dataset, _dataset_name = ood_data.split(".")
        try:
            if _dataset_name == "default":
                load_dataset(_dataset)
            else:
                load_dataset(_dataset, _dataset_name)
        except FileNotFoundError:
            org = _dataset.split("_")[0]
            _dataset_main = '_'.join(_dataset.split("_")[1:])
            _dataset = f"{org}/{_dataset_main}"
            try:
                if _dataset_name == "default":
                    load_dataset(_dataset)
                else:
                    load_dataset(_dataset, _dataset_name)
            except FileNotFoundError:
                raise ValueError(f"dataset {_dataset} is not found")
        with open(i) as f:
            metric = json.load(f)
            metrics_ood.append([_dataset, _dataset_name, metric])
    metrics = '\n'.join([format_metric(d, t, m) for d, t, m in [metric_main] + metrics_ood])
    # readme table
    link = f'https://huggingface.co/{model_name}/raw/main/eval/{eval_file}.{dataset.replace("/", "_")}.{dataset_name}.json'
    markdown_table = f"""
### Metrics

| Dataset | Type | BLEU4 | ROUGE-L | METEOR | BERTScore | MoverScore | Link |
|:--------|:-----|------:|--------:|-------:|----------:|-----------:|-----:|
| [{metric_main[0]}](https://huggingface.co/datasets/{metric_main[0]}) | {metric_main[1]} | {round(metric_main[2]['test']['Bleu_4'], 3)} | {round(metric_main[2]['test']['ROUGE_L'], 3)} | {round(metric_main[2]['test']['METEOR'], 3)} | {round(metric_main[2]['test']['BERTScore'], 3)} | {round(metric_main[2]['test']['MoverScore'], 3)} | [link]({link}) | 
"""
    if len(metrics_ood) != 0:
        content = "\n".join([
                      f"| [{d}](https://huggingface.co/datasets/{d}) | {t} | {round(m['test']['Bleu_4'], 3)} | {round(m['test']['ROUGE_L'], 3)} | {round(m['test']['METEOR'], 3)} | {round(m['test']['BERTScore'], 3)} | {round(m['test']['MoverScore'], 3)} | "
                      f"[link](https://huggingface.co/{model_name}/raw/main/eval_ood/{eval_file}.{d.replace('/', '_')}.{t}.json) |"
                      for d, t, m in metrics_ood])
        markdown_table_ood = f"""
### Out-of-domain Metrics
        
| Dataset | Type | BLEU4 | ROUGE-L | METEOR | BERTScore | MoverScore | Link |
|:--------|:-----|------:|--------:|-------:|----------:|-----------:|-----:|
{content}
"""
    else:
        markdown_table_ood = ''

    return f"""
---
license: cc-by-4.0
metrics:
- bleu4
- meteor
- rouge-l
- bertscore
- moverscore
language: {la}
datasets:
- {dataset}
pipeline_tag: text2text-generation
tags:
{tags}
widget:
{widget}
model-index:
- name: {model_name}
  results:
{metrics}
---

# Language Models Fine-tuning on Question Generation: `{model_name}`
This model is fine-tuned version of [{language_model}](https://huggingface.co/{language_model}) for question generation task on the 
[{dataset}](https://huggingface.co/datasets/{dataset}) (dataset_name: {dataset_name}).
{add_info}

### Overview
- **Language model:** [{language_model}](https://huggingface.co/{language_model})   
- **Language:** {la}  
- **Training data:** [{dataset}](https://huggingface.co/datasets/{dataset}) ({dataset_name})
- **Online Demo:** [https://autoqg.net/](https://autoqg.net/)
- **Repository:** [https://github.com/asahi417/lm-question-generation](https://github.com/asahi417/lm-question-generation)
- **Paper:** [TBA](TBA)

### Usage
```python
{usage}
```

## Evaluation Metrics

{markdown_table}

{markdown_table_ood}

## Training hyperparameters

The following hyperparameters were used during fine-tuning:
{config_text}

The full configuration can be found at [fine-tuning config file](https://huggingface.co/{model_name}/raw/main/trainer_config.json).

## Citation
TBA
"""
