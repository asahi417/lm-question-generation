import os
import json
import re
from os.path import join as pj
from glob import glob
import pandas as pd
from lmqg.language_model import TASK_PREFIX
from datasets import load_dataset

bib = """@inproceedings{ushio-etal-2022-generative,
    title = "{G}enerative {L}anguage {M}odels for {P}aragraph-{L}evel {Q}uestion {G}eneration",
    author = "Ushio, Asahi  and
        Alva-Manchego, Fernando  and
        Camacho-Collados, Jose",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, U.A.E.",
    publisher = "Association for Computational Linguistics",
}
"""
paper_link = "[https://arxiv.org/abs/2210.03992](https://arxiv.org/abs/2210.03992)"
version_description = {
    'default': "This model is fine-tuned without parameter search (default configuration is taken from [ERNIE-GEN](https://arxiv.org/abs/2001.11314)).",
    'no-answer': "This model is fine-tuned without answer information, i.e. generate a question only given a paragraph (note that normal model is fine-tuned to generate a question given a pargraph and an associated answer in the paragraph).",
    'no-paragraph': "This model is fine-tuned without pargraph information but only the sentence that contains the answer.",
}
sample_qa_dict = {
    "en": [
        "question: What is a person called is practicing heresy?, context: Heresy is any provocative belief or theory that is strongly at variance with established beliefs or customs. A heretic is a proponent of such claims or beliefs. Heresy is distinct from both apostasy, which is the explicit renunciation of one's religion, principles or cause, and blasphemy, which is an impious utterance or action concerning God or sacred things.",
        "question: who created the post as we know it today?, context: 'So much of The Post is Ben,' Mrs. Graham said in 1994, three years after Bradlee retired as editor. 'He created it as we know it today.'— Ed O'Keefe (@edatpost) October 21, 2014"],
    "ja": [
        "question: 新型車両として6000系が構想されたのは、製造費用のほか、どんな費用を抑えるためだったの?, context: 三多摩地区開発による沿線人口の増加、相模原線延伸による多摩ニュータウン乗り入れ、都営地下鉄10号線(現都営地下鉄新宿線、以下新宿線と表記する)乗入構想により、京王線の利用客増加が見込まれ、相当数の車両を準備する必要に迫られるなか、製造費用、保守費用を抑えた新型車両として6000系が構想された。新宿線建設に際してはすでに1号線(後の浅草線)を1,435mm軌間で開業させていた東京都は京成電鉄と1号線との乗り入れにあたり京成電鉄の路線を1,372mmから1,435mmに改軌させた事例や、1,372mm軌間の特殊性から運輸省(当時、2001年から国土交通省)と共に京王にも改軌を求めたが、改軌工事中の輸送力確保が困難なことを理由に改軌しないことで決着している。",
        "question: 1968年に開催されたオリンピックの名前は何ですか?, context: オリンピックが世界的大イベントに成長するに従って政治に左右されるようになると、1968年のメキシコシティ大会では黒人差別を訴える場と化し、1972年のミュンヘン大会ではアラブのゲリラによるイスラエル選手に対するテロ事件まで起きた(ミュンヘンオリンピック事件)。1976年のモントリオール大会になると、ニュージーランドのラグビーチームの南アフリカ遠征に反対してアフリカの諸国22ヶ国がボイコットを行った。そして、1980年のモスクワ大会ではソ連のアフガニスタン侵攻に反発したアメリカ・西ドイツ・日本などの西側諸国が相次いでボイコットを行った。1984年ロサンゼルス大会ではソ連と東側諸国が報復ボイコットを行ない、参加したのはソ連と対立していた中国とルーマニアだけだった。中でも、イラン革命後のイラン・イスラム共和国はモスクワとロサンゼルス双方のオリンピックをボイコットしている。オリンピックが巨大化するに従って財政負担の増大が大きな問題となり、1976年の夏季大会では大幅な赤字を出し、その後夏季・冬季とも立候補都市が1〜2都市だけという状態が続いた。"
    ],
    "ru": [
        "question: чем соответствует абсолютная погрешность скорости света ?, context: Наивысшая точность измерений была достигнута в начале 1970-х. В 1975 году XV Генеральная конференция по мерам и весам зафиксировала это положение и рекомендовала считать скорость света, равной 299 792 458 м/с с относительной погрешностью 4•10−9, что соответствует абсолютной погрешности 1,1 м/с. Впоследствии это значение скорости света было положено в основу определения метра в Международной системе единиц (СИ), а сама скорость света стала рассматриваться как фундаментальная физическая постоянная, по определению равная указанному значению точно.",
        "question: Какие начинания предпринял Lloyds в начале 1970-х годов?, context: В начале 1970-х Lloyds начал расширять деятельность на международной арене, для чего был создан Lloyds Bank International. География его деятельности включала ФРГ, Швейцарию, Ближний Восток, Австралию, Канаду и США; к 1978 году Lloyds был представлен в 43 странах. В 1972 году было создано подразделение страхования, а в 1973 году была основана лизинговая компания Lloyds Leasing. В 1979 году банк начал предоставлять услуги ипотечного кредитования (при покупке недвижимости стоимостью от £25 000 до £150 000). В 1982 году начало работу агентство недвижимости Blackhorse Agencies, к 1989 году у него было 563 отделения. В 1986 году сфера деятельности Lloyds Bank PLC ещё больше расширилась с учреждением брокерской конторы и торгового банка Lloyds Merchant Bank. В 1988 году была поглощена страховая компания Abbey Life Group PLC; после объединения с ней всей своей страховой деятельности была образована дочерняя компания Lloyds Abbey Life. В 1995 году Lloyds Bank Plc объединился с TSB Group plc (группой, образованной в 1986 году из четырёх сберегательных банков Trustee Savings Banks) под названием Lloyds TSB Bank plc. В 2000 году за £7 млрд была поглощена шотландская взаимная страховая компания Scottish Widows."
    ],
    "ko": [
        "question: 매드 클라운이 참가해 큰 화제를 모았던 프로그램은?, context: 과거 소울 컴퍼니 소속으로 소울 컴퍼니 해체 후 현재의 소속사는 스타쉽 엑스이다. Mad Clown vs Crucial Star (매드 클라운 vs 크루셜 스타)라는 프로젝트 그룹으로 크루셜 스타와 함께 활동하기도 하였으며, 2013년부터는 MC인 저스디스와 팀을 이루어 랩 듀오 커먼콜드로 활동하고 있다. 또한 Mnet 《쇼미더머니 2》에서 참가자로 참가하여 큰 화제를 모았으며, 《쇼미더머니 5》에서는 길 & 매드 클라운 팀으로 프로듀서로 출연하였다., 재발매 물량도 완판되어 추가 제작에 들어갔다. 2016년 4월, 소속사와 자신의 SNS를 통해 2016년 5월 15일 현재 교제 중인 일반인 여자친구와의 결혼을 공식발표하였다.",
        "question: 1913년 필라델피아 애슬레틱스의 개막전 상대는?, context: 1913년 시즌을 앞두고 스프링 트레이닝에서 잭 쿰스는 앨라배마 주 몽고메리에서 고열로 힘들어했는데, 당시에는 식중독 및 늑막염 진단을 받고 휴식을 취했다. 4월 10일, 보스턴 레드삭스를 상대로 치러진 개막전에서 잭 쿰스는 선발투수로 내정되었다. 그는 3이닝을 노히트로 막고 6회 치프 벤더와 교체되었으며, 경기는 10-5로 애슬레틱스가 승리했다. 이틀 뒤에 다시 선발 등판에 나섰으나 ⁄3이닝 동안 2피안타 1볼넷, 4실점만을 기록하고 강판되었다. 쿰스는 보스턴에서의 시리즈를 끝내고 팀 동료들과 함께 워싱턴으로 향했지만, 고통이 심해지자 구단은 그를 필라델피아로 돌려보냈다. 그곳에서 그는 장티푸스 진단을 받고 휴식을 취했으며, 8월에 다시 팀에 복귀하려고 했지만 정상적인 회복을 위해서 다시 병원에 들어갔다. 이 기간 몸무게가 25 kg 가량이나 감소했다. 이 해 필라델피아 애슬레틱스는 월드 시리즈에서 2년만에 다시 뉴욕 자이언츠와 맞붙었고, 우승을 차지했다. 쿰스의 공백기는 다음해인 1914년 시즌까지 길어졌다. 이 해 시즌에는 팀 순위가 정해진 시즌 막판에야 두 경기에 선발 출전해서, 도합 8이닝 8피안타 4실점, 4.50의 평균자책점을 기록했다. 시즌 후인 12월 9일, 애슬레틱스에서 방출되었다."
    ],
    "es": [
        "question: ¿Cuál es la población de Nueva York a partir de 2014?, context: Situada en uno de los mayores puertos naturales del mundo, la ciudad de Nueva York consta de cinco municipios, cada uno de los cuales es un condado separado del estado de Nueva York. Los cinco distritos - Brooklyn, Queens, Manhattan, el Bronx y Staten Island - se consolidaron en una sola ciudad en 1898. Con una población censada estimada en 2014 de 8.491.079 habitantes distribuidos en una superficie de solo 790 km ², Nueva York es la ciudad más densamente poblada de los Estados Unidos. Hasta 800 idiomas se hablan en Nueva York, por lo que es la ciudad más lingüísticamente diversa del mundo. Según estimaciones del censo de 2014, la región metropolitana de la ciudad de Nueva York sigue siendo por un margen significativo la más poblada de los Estados Unidos, según lo definido tanto por el Área Estadística Metropolitana (20,1 millones de residentes). En 2013, el MSA produjo un producto metropolitano bruto (GMP) de casi US $1,39 billones, mientras que en 2012, el CSA generó un GMP de más de US $1,55 billones, ambos clasificados en primer lugar.",
        "question: ¿Cómo se llama el ejército personal de Sassou?, context: El progreso democrático del Congo se descarriló en 1997, cuando Lissouba y Sassou comenzaron a luchar por el poder en la guerra civil. A medida que se acercaban las elecciones presidenciales de julio de 1997, las tensiones entre los campos de Lissouba y Sassou aumentaron. El 5 de junio, las fuerzas del gobierno del presidente Lissouba rodearon el complejo de Sassou en Brazzaville y Sassou ordenó a los miembros de su milicia privada (conocida como Cobras) resistir. Así comenzó un conflicto de cuatro meses que destruyó o dañó gran parte de Brazzaville y causó decenas de miles de muertes civiles. A principios de octubre, el régimen socialista angoleño comenzó una invasión del Congo para instalar a Sassou en el poder. A mediados de octubre, el gobierno de Lissouba cayó. Poco después, Sassou se declaró presidente."
    ],
    "fr": [
        "question: En quelle année a-t-on trouvé trace d'un haut fourneau similaire?, context: Cette technologie ne disparaît qu'au début du XXe siècle. On retrouve vers 1900 un haut fourneau similaire dans le Bulacan, aux Philippines. Plus tard encore, le « haut fourneau dans la cour » prôné par Mao Zedong pendant le Grand Bond en avant est de ce type. L'expérience n'est un échec technique que dans les régions où le savoir-faire n'existe pas, ou a disparu.",
        "question: Comment appelle-t-on la Guerre de 14-18 ?, context: Ce black dog peut être lié à des évènements traumatisants issus du monde extérieur, tels que son renvoi de l'Amirauté après la catastrophe des Dardanelles, lors de la Grande Guerre de 14-18, ou son rejet par l'électorat en juillet 1945. On sait également que dans ces deux cas, la guérison, certes lente et douloureuse et jamais complète ni définitive, se fera grâce à la peinture. D'un autre côté, étant donnés les symptômes de ce mal que Churchill éprouvait de plus en plus, il ne pouvait rien moins qu'être purement associé à de telles causes extrinsèques, ce qui correspond au profil classique de la dépression majeure unipolaire ou bipolaire."

    ],
    "de": [
        "question: Welche Auszeichnung hat die Wartburg 1999 erhalten?, context: Thüringen == Kultur == Die Kulturlandschaft Thüringens ist bedingt durch die lange politische Zersplitterung (bis 1920) recht vielfältig. Diese Vielfalt hat sich bis heute erhalten und findet in den verschiedenen ehemaligen Residenzen im Land mit ihren historisch gewachsenen Museen und Theatern Ausdruck. Parallel zur Vielfalt der Landesteile verbinden aber vor allem die ähnliche Küche sowie ähnlichen Feste und Bräuche. Prägend für die Kultur sind nach wie vor die zahlreichen Stätten der klassischen Hochkultur von der Reformation bis zum Bauhaus hinter denen die Orte der Gegenwartskultur ein Stück weit zurückfallen. Zum UNESCO-Welterbe in Thüringen gehören seit 1996 die Bauhaus-Stätten in Weimar mit dem zwischen 1904 und 1911 nach Plänen von Henry van de Velde errichteten Hauptgebäude der Bauhaus-Universität, der Kunstgewerbeschule Weimar und dem Musterhaus Am Horn, seit 1998 die elf Stätten des Klassischen Weimars (Goethes Wohnhaus, Schillers Wohnhaus, Herderkirche und Herder-Stätten, Weimarer Stadtschloss, Wittumspalais, Herzogin Anna Amalia Bibliothek, Park an der Ilm mit Goethes Gartenhaus und Römischem Haus, Schloss Belvedere, Schloss Ettersburg, Schloss Tiefurt, Historischer Friedhof Weimar), seit 1999 die Wartburg bei Eisenach und seit 2011 der Nationalpark Hainich als Teil der Europäischen Buchenurwälder." ,
        "question: Wann endete die Aberdeen Regierung? , context: Krimkrieg === Großbritannien === Der Krimkrieg zeigte, dass es erhebliche Missstände im britischen Militär gab. Dadurch verlor die Regierung Aberdeen erheblich an Ansehen. Im Februar 1855 wurde sie zum Rücktritt gezwungen, und Palmerston übernahm die Bildung eines neuen Kabinetts. Der spätere britische Premierminister Disraeli erklärte den Krieg aus einer von Südasien eingenommenen Perspektive zu einem „indischen Krieg“, da es zuvor (irreale) Befürchtungen gegeben hatte, dass Russland durch eine Expansion nach Süden das britische Indien in Gefahr bringen könnte. Das Verhältnis zwischen Großbritannien und Russland blieb bis ins 20. Jahrhundert aus ideologischen und weltmachtpolitischen Gründen angespannt. Der Krieg führte in Großbritannien zur Bildung eines modernen Nationalmythos des die Ehre der Nation verteidigenden „gemeinen“ Soldaten, anstelle des Aristokraten früherer Kriege. In der Mittelklasse kam es zu einem neuen Gefühl des Selbstbewusstseins im Zusammenhang von Ideen wie professioneller Fähigkeit und dem Leistungsprinzip. Die Mittelklasse erkannte sich in einer Florence Nightingale wieder, die zur Nationalheldin aufstieg. Die Königin stiftete 1857 das Victoria-Kreuz, mit dem erstmals Nichtoffiziere ausgezeichnet werden konnten."
    ],
    "it": [
        "question: Quale batterio ha il nome del paese che colpisce di più nel suo nome?, context: Il complesso M. tubercolosi (MTBC) comprende altri quattro micobatteri causa di tubercolosi: M. bovis, M. africanum, M. canetti e M. microti. M. africanum non è molto diffuso, ma è una causa significativa di tubercolosi in alcune parti dell' Africa. M. bovis era una volta una causa comune della tubercolosi, ma l' introduzione del latte pastorizzato ha quasi completamente eliminato questo problema di salute pubblica nei paesi sviluppati. M. canetti è raro e sembra essere limitato al Corno d' Africa, anche se alcuni casi sono stati osservati negli emigranti africani. M. microti è anche raro ed è visto quasi solo in persone immunodeficienti, anche se la sua prevalenza può essere significativamente sottovalutata.",
    ],
    "zh": [
        "question: 哪个火车站离警察局大楼很近？, context: 南安普敦的警察服务由汉普郡警察提供。南安普敦行动的主要基地是一座新的八层专用建筑，造价3000万英镑。该建筑位于南路，2011年启用，靠近南安普敦中央火车站。此前，南安普顿市中心的行动位于市民中心西翼，但由于设施老化，加上计划在旧警察局和地方法院建造一座新博物馆，因此必须搬迁。在Portswood、Banister Park、Hille和Shirley还有其他警察局，在南安普顿中央火车站还有一个英国交通警察局。",
    ]
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
    ],
    "zh": [
        '南安普敦的警察服务由汉普郡警察提供。南安普敦行动的主要基地是一座新的八层专用建筑，造价3000万英镑。该建筑位于南路，2011年启用，靠近<hl> 南安普敦中央 <hl>火车站。此前，南安普顿市中心的行动位于市民中心西翼，但由于设施老化，加上计划在旧警察局和地方法院建造一座新博物馆，因此必须搬迁。在Portswood、Banister Park、Hille和Shirley还有其他警察局，在南安普顿中央火车站还有一个英国交通警察局。',
        '芝加哥大学的<hl> 1960—61 <hl>集团理论年汇集了Daniel Gorenstein、John G. Thompson和Walter Feit等团体理论家，奠定了一个合作的基础，借助于其他众多数学家的输入，1982中对所有有限的简单群进行了分类。这个项目的规模超过了以往的数学研究，无论是证明的长度还是研究人员的数量。目前正在进行研究，以简化这一分类的证明。如今，群论仍然是一个非常活跃的数学分支，影响着许多其他领域'
    ]
}
sample_ae_dict = {
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
    ],
    "zh": [
        '南安普敦的警察服务由汉普郡警察提供。 南安普敦行动的主要基地是一座新的八层专用建筑，造价3000万英镑。 <hl> 该建筑位于南路，2011年启用，靠近 南安普敦中央 火车站。 <hl> 此前，南安普顿市中心的行动位于市民中心西翼，但由于设施老化，加上计划在旧警察局和地方法院建造一座新博物馆，因此必须搬迁。 在Portswood、Banister Park、Hille和Shirley还有其他警察局，在南安普顿中央火车站还有一个英国交通警察局。'
    ]
}
sample_lmqg_dict = {
    "en": ["William Turner was an English painter who specialised in watercolour landscapes", "William Turner"],
    "ja": ["フェルメールの作品では、17世紀のオランダの画家、ヨハネス・フェルメールの作品について記述する。フェルメールの作品は、疑問作も含め30数点しか現存しない。現存作品はすべて油彩画で、版画、下絵、素描などは残っていない。", "30数点"],
    "ru": ["Нелишним будет отметить, что, развивая это направление, Д. И. Менделеев, поначалу априорно выдвинув идею о температуре, при которой высота мениска будет нулевой, в мае 1860 года провёл серию опытов.", "в мае 1860 года"],
    "ko": ["1990년 영화 《 남부군 》에서 단역으로 영화배우 첫 데뷔에 이어 같은 해 KBS 드라마 《지구인》에서 단역으로 출연하였고 이듬해 MBC 《여명의 눈동자》를 통해 단역으로 출연하였다.", "남부군"],
    "es": ["a noviembre , que es también la estación lluviosa.", "noviembre"],
    "fr": ["Créateur » (Maker), lui aussi au singulier, « le Suprême Berger » (The Great Shepherd) ; de l'autre, des réminiscences de la théologie de l'Antiquité : le tonnerre, voix de Jupiter, « Et souvent ta voix gronde en un tonnerre terrifiant », etc.", "le Suprême Berger"],
    "de": ["das erste weltweit errichtete Hermann Brehmer 1855 im niederschlesischen ''Görbersdorf'' (heute Sokołowsko, Polen).", "1855"],
    "it": ["Dopo il 1971 , l' OPEC ha tardato ad adeguare i prezzi per riflettere tale deprezzamento.", "Dopo il 1971"],
    "zh": ['南安普敦的警察服务由汉普郡警察提供。南安普敦行动的主要基地是一座新的八层专用建筑，造价3000万英镑。该建筑位于南路，2011年启用，靠近南安普敦中央火车站。此前，南安普顿市中心的行动位于市民中心西翼，但由于设施老化，加上计划在旧警察局和地方法院建造一座新博物馆，因此必须搬迁。在Portswood、Banister Park、Hille和Shirley还有其他警察局，在南安普顿中央火车站还有一个英国交通警察局。', '南安普敦中央']
}
language_dict = {
    "qg_frquad": 'fr',
    "qg_itquad": 'it',
    "qg_koquad": 'ko',
    "qg_dequad": 'de',
    "qg_jaquad": 'ja',
    "qg_ruquad": 'ru',
    "qg_esquad": 'es',
    "qg_zhquad": 'zh',
    "qag_frquad": 'fr',
    "qag_itquad": 'it',
    "qag_koquad": 'ko',
    "qag_dequad": 'de',
    "qag_jaquad": 'ja',
    "qag_ruquad": 'ru',
    "qag_esquad": 'es',
    "qag_zhquad": 'zh',
}


def keep_qag_metric(df):
    df = df.T
    col = df.columns
    for i in ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "BERTScore", "MoverScore"]:
        if i in col:
            df.pop(i)
    return df.T


def __format_metric(metric, metric_label, metric_label_type, is_multitask, is_end2end):
    tmp = ""
    if "QAAlignedF1Score (BERTScore)" not in metric['test']:
        if "Bleu_4" in metric["test"]:
            tmp += f"""
    - name: BLEU4 ({metric_label})
      type: bleu4_{metric_label_type}
      value: {metric["test"]["Bleu_4"]}"""
        if "ROUGE_L" in metric["test"]:
            tmp += f"""
    - name: ROUGE-L ({metric_label})
      type: rouge_l_{metric_label_type}
      value: {metric["test"]["ROUGE_L"]}"""
        if "METEOR" in metric["test"]:
            tmp += f"""
    - name: METEOR ({metric_label})
      type: meteor_{metric_label_type}
      value: {metric["test"]["METEOR"]}"""
        if "BERTScore" in metric["test"]:
            tmp += f"""
    - name: BERTScore ({metric_label})
      type: bertscore_{metric_label_type}
      value: {metric["test"]["BERTScore"]}"""
        if "MoverScore" in metric["test"]:
            tmp += f"""
    - name: MoverScore ({metric_label})
      type: moverscore_{metric_label_type}
      value: {metric["test"]["MoverScore"]}"""
        if "AnswerF1Score" in metric['test']:
            tmp += f"""
    - name: AnswerF1Score ({metric_label})
      type: answer_f1_score__{metric_label_type}
      value: {metric["test"]["AnswerF1Score"]}"""
        if "AnswerExactMatch" in metric['test']:
            tmp += f"""
    - name: AnswerExactMatch ({metric_label})
      type: answer_exact_match_{metric_label_type}
      value: {metric["test"]["AnswerExactMatch"]}"""
    if "QAAlignedF1Score (BERTScore)" in metric['test']:
        tmp += f"""
    - name: QAAlignedF1Score-BERTScore ({metric_label}){"" if is_multitask or is_end2end else " [Gold Answer]"}
      type: qa_aligned_f1_score_bertscore_{metric_label_type}{"" if is_multitask or is_end2end else "_gold_answer"}
      value: {metric["test"]["QAAlignedF1Score (BERTScore)"]}"""
    if "QAAlignedRecall (BERTScore)" in metric['test']:
        tmp += f"""
    - name: QAAlignedRecall-BERTScore ({metric_label}){"" if is_multitask or is_end2end else " [Gold Answer]"}
      type: qa_aligned_recall_bertscore_{metric_label_type}{"" if is_multitask or is_end2end else "_gold_answer"}
      value: {metric["test"]["QAAlignedRecall (BERTScore)"]}"""
    if "QAAlignedPrecision (BERTScore)" in metric['test']:
        tmp += f"""
    - name: QAAlignedPrecision-BERTScore ({metric_label}){"" if is_multitask or is_end2end else " [Gold Answer]"}
      type: qa_aligned_precision_bertscore_{metric_label_type}{"" if is_multitask or is_end2end else "_gold_answer"}
      value: {metric["test"]["QAAlignedPrecision (BERTScore)"]}"""
    if "QAAlignedF1Score (MoverScore)" in metric['test']:
        tmp += f"""
    - name: QAAlignedF1Score-MoverScore ({metric_label}){"" if is_multitask or is_end2end else " [Gold Answer]"}
      type: qa_aligned_f1_score_moverscore_{metric_label_type}{"" if is_multitask or is_end2end else "_gold_answer"}
      value: {metric["test"]["QAAlignedF1Score (MoverScore)"]}"""
    if "QAAlignedRecall (MoverScore)" in metric['test']:
        tmp += f"""
    - name: QAAlignedRecall-MoverScore ({metric_label}){"" if is_multitask or is_end2end else " [Gold Answer]"}
      type: qa_aligned_recall_moverscore_{metric_label_type}{"" if is_multitask or is_end2end else "_gold_answer"}
      value: {metric["test"]["QAAlignedRecall (MoverScore)"]}"""
    if "QAAlignedPrecision (MoverScore)" in metric['test']:
        tmp += f"""
    - name: QAAlignedPrecision-MoverScore ({metric_label}){"" if is_multitask or is_end2end else " [Gold Answer]"}
      type: qa_aligned_precision_moverscore_{metric_label_type}{"" if is_multitask or is_end2end else "_gold_answer"}
      value: {metric["test"]["QAAlignedPrecision (MoverScore)"]}"""
    return tmp


def format_metric(dataset, dataset_type, metric, metric_qag, metric_qa, metric_ae, metric_qag_pipe, is_multitask, is_end2end, is_qa, is_ae):
    metric_label = 'Question Generation'
    metric_label_type = "question_generation"
    if is_qa:
        metric_label = 'Question Answering'
        metric_label_type = "question_answering"
    elif is_ae:
        metric_label = 'Answer Extraction'
        metric_label_type = "answer_extraction"
    elif is_end2end:
        metric_label = 'Question & Answer Generation'
        metric_label_type = "question_answer_generation"
    tmp = f"""  - task:
      name: Text2text Generation
      type: text2text-generation
    dataset:
      name: {dataset}
      type: {dataset_type}
      args: {dataset_type}
    metrics:"""
    tmp += __format_metric(metric, metric_label, metric_label_type, is_multitask, is_end2end)
    if metric_qag is not None:
        tmp += __format_metric(metric_qag, 'Question & Answer Generation (with Gold Answer)', "question_answer_generation_with_gold_answer", is_multitask, is_end2end)
    if metric_qa:
        tmp += __format_metric(metric_qa, 'Question Answering', "question_answering", is_multitask, is_end2end)
    if metric_ae:
        tmp += __format_metric(metric_ae, 'Answer Extraction', "answer_extraction", is_multitask, is_end2end)
    if metric_qag_pipe:
        tmp += __format_metric(metric_qag_pipe, 'Question & Answer Generation', "question_answer_generation", is_multitask, is_end2end)
    return tmp


def format_usage(model_name, sample, sample_ae):
    if sample_ae is not None:
        return f"""from transformers import pipeline

pipe = pipeline("text2text-generation", "{model_name}")

# answer extraction
answer = pipe("{sample[0]}")

# question generation
question = pipe("{sample_ae[0]}")
"""
    else:
        return f"""from transformers import pipeline

pipe = pipeline("text2text-generation", "{model_name}")
output = pipe("{sample[0]}")
"""


def format_usage_lmqg(model_name, language, is_multitask, is_end2end, is_qa, is_ae):
    desc = f"""from lmqg import TransformersQG

# initialize model
model = TransformersQG(language="{language}", model="{model_name}")
"""
    if is_multitask or is_end2end:
        desc += f"""
# model prediction
question_answer_pairs = model.generate_qa("{sample_lmqg_dict[language][0]}")
"""
    elif is_qa:
        _tmp = sample_qa_dict[language][0]
        __q, __c = _tmp.split(', context:')
        __q = __q.replace("question: ", '')
        desc += f"""
# model prediction
answers = model.answer_q(list_question="{__q}", list_context="{__c}")
"""
    elif is_ae:
        desc += f"""
# model prediction
answers = model.generate_a("{sample_lmqg_dict[language][0]}")
"""
    else:
        desc += f"""
# model prediction
questions = model.generate_q(list_context="{sample_lmqg_dict[language][0]}", list_answer="{sample_lmqg_dict[language][1]}")
"""
    return desc


def get_readme(model_name: str, model_checkpoint: str):
    with open(pj(model_checkpoint, "trainer_config.json")) as f:
        config = json.load(f)
    config_text = "\n".join([f" - {k}: {v}" for k, v in config.items()])
    language_model = config['model']
    dataset = config['dataset_path']
    prefix_types = config['prefix_types']
    dataset_name = config['dataset_name']
    dataset_alias = os.path.basename(dataset)
    la = language_dict[dataset_alias] if dataset_alias in language_dict else 'en'
    header = f"This model is fine-tuned version of [{language_model}](https://huggingface.co/{language_model}) for question generation task on the [{dataset}](https://huggingface.co/datasets/{dataset}) (dataset_name: {dataset_name}) via [`lmqg`](https://github.com/asahi417/lm-question-generation)."
    metric_title = "Question Generation"
    # model_version
    eval_file = "metric.first.sentence.paragraph_answer.question"
    eval_file_qag = "metric.first.answer.paragraph.questions_answers"
    eval_file_qa = "metric.first.answer.paragraph_question.answer"
    eval_file_ae = "metric.first.answer.paragraph_sentence.answer"
    add_info = []
    _sample = sample_qg_dict[la]
    _is_qag = False
    _is_qa = False
    _is_ae = False
    _is_multitask = False

    # Multitask QAG Models
    if model_name.endswith('multitask') or '-qg-ae' in model_name:
        _is_multitask = True
        header = f"This model is fine-tuned version of [{language_model}](https://huggingface.co/{language_model}) for question generation and answer extraction jointly on the [{dataset}](https://huggingface.co/datasets/{dataset}) (dataset_name: {dataset_name}) via [`lmqg`](https://github.com/asahi417/lm-question-generation)."
    # E2E QAG Models
    elif 'qag' in model_name.split('-'):
        _sample = [re.sub(r'\s+', ' ', _sample[0].replace('<hl>', ''))]
        eval_file = "metric.first.answer.paragraph.questions_answers"
        metric_title = "Question & Answer Generation"
        eval_file_qag = eval_file_ae = None
        header = f"This model is fine-tuned version of [{language_model}](https://huggingface.co/{language_model}) for question & answer pair generation task on the [{dataset}](https://huggingface.co/datasets/{dataset}) (dataset_name: {dataset_name}) via [`lmqg`](https://github.com/asahi417/lm-question-generation)."
        _is_qag = True
        if 'np' in model_name.split('-'):
            add_info.append("This model is fine-tuned without a task prefix.")
    # QA Models
    elif "question-answering" in model_name or 'qa' in model_name.split('-'):
        _sample = sample_qa_dict[la]
        eval_file = "metric.first.answer.paragraph_question.answer"
        metric_title = "Question Answering"
        eval_file_qa = eval_file_qag = eval_file_ae = None
        header = f"This model is fine-tuned version of [{language_model}](https://huggingface.co/{language_model}) for question answering task on the [{dataset}](https://huggingface.co/datasets/{dataset}) (dataset_name: {dataset_name}) via [`lmqg`](https://github.com/asahi417/lm-question-generation)."
        _is_qa = True
    # Answer Extraction Models
    elif "answer-extraction" in model_name or ('ae' in model_name.split('-') and 'qg' not in model_name.split('-')):
        _sample = sample_ae_dict[la]
        eval_file = "metric.first.answer.paragraph_sentence.answer"
        metric_title = "Answer Extraction"
        eval_file_qa = eval_file_qag = eval_file_ae = None
        header = f"This model is fine-tuned version of [{language_model}](https://huggingface.co/{language_model}) for answer extraction on the [{dataset}](https://huggingface.co/datasets/{dataset}) (dataset_name: {dataset_name}) via [`lmqg`](https://github.com/asahi417/lm-question-generation)."
        _is_ae = True
    # QG Models
    else:
        if model_name.endswith('no-answer'):
            _sample = ["<hl> " + re.sub(r'\s+', ' ', i.replace('<hl>', '')) + " <hl>" for i in _sample]
            add_info.append(version_description['no-answer'])
            eval_file = "metric.first.sentence.paragraph_sentence.question"
        elif model_name.endswith('no-paragraph'):
            add_info.append(version_description['no-paragraph'])
            eval_file = "metric.first.sentence.sentence_answer.question"
        elif model_name.endswith('default'):
            add_info.append(version_description['default'])

    _sample = [re.sub(r"\A\s+", "", i) for i in _sample]
    add_info = ' '.join(add_info)

    # get widget
    _sample_ae = None
    if _is_qag:
        tags = "- questions and answers generation"
        _sample = _sample if prefix_types is None else [f'{TASK_PREFIX["qag"]}: {i}' for i in _sample]
        widget = '\n'.join([f"""- text: "{i}"\n  example_title: "Questions & Answers Generation Example {n + 1}" """ for n, i in enumerate(_sample)])
    elif _is_qa:
        tags = "- question answering"
        _sample = _sample if prefix_types is None else [f'{TASK_PREFIX["qa"]}: {i}' for i in _sample]
        widget = '\n'.join([f"""- text: "{i}"\n  example_title: "Question Answering Example {n + 1}" """ for n, i in enumerate(_sample)])
    elif _is_ae:
        tags = "- answer extraction"
        _sample = _sample if prefix_types is None else [f'{TASK_PREFIX["ae"]}: {i}' for i in _sample]
        widget = '\n'.join([f"""- text: "{i}"\n  example_title: "Answering Extraction Example {n + 1}" """ for n, i in enumerate(_sample)])
    else:
        if _is_multitask:  # multitask
            tags = "- question generation\n- answer extraction"
            _sample = [f'{TASK_PREFIX["qg"]}: {i}' for i in _sample]
            _sample_ae = [f'{TASK_PREFIX["ae"]}: {i}' for i in sample_ae_dict[la]]
            widget = '\n'.join([f"""- text: "{i}"\n  example_title: "Question Generation Example {n + 1}" """ for n, i in enumerate(_sample)])
            widget += '\n' + '\n'.join([f"""- text: "{i}"\n  example_title: "Answer Extraction Example {n + 1}" """ for n, i in enumerate(_sample_ae)])
        else:
            tags = "- question generation"
            _sample = _sample if prefix_types is None else [f'{TASK_PREFIX["qg"]}: {i}' for i in _sample]
            widget = '\n'.join([f"""- text: "{i}"\n  example_title: "Question Generation Example {n + 1}" """ for n, i in enumerate(_sample)])

    # usage
    usage = format_usage(model_name, _sample, _sample_ae)
    usage_lmqg = format_usage_lmqg(model_name, la, _is_multitask, _is_qag, _is_qa, _is_ae)

    # metric
    with open(pj(model_checkpoint, "eval", f"{eval_file}.{dataset.replace('/', '_')}.{dataset_name}.json")) as f:
        metric = {k: {_k: round(_v * 100 if _k not in ["AnswerF1Score", "AnswerExactMatch"] else _v, 2) for _k, _v in v.items()} for k, v in json.load(f).items()}

    metric_qag = None
    if eval_file_qag is not None:
        tmp_path = pj(model_checkpoint, "eval", f"{eval_file_qag}.{dataset.replace('/', '_')}.{dataset_name}.json")
        if os.path.exists(tmp_path):
            with open(tmp_path) as f:
                metric_qag = {k: {_k: round(_v * 100 if _k not in ["AnswerF1Score", "AnswerExactMatch"] else _v, 2) for _k, _v in v.items()} for k, v in json.load(f).items()}

    metric_qa = None
    if eval_file_qa is not None:
        tmp_path = pj(model_checkpoint, "eval", f"{eval_file_qa}.{dataset.replace('/', '_')}.{dataset_name}.json")
        if os.path.exists(tmp_path):
            with open(tmp_path) as f:
                metric_qa = {k: {_k: round(_v * 100 if _k not in ["AnswerF1Score", "AnswerExactMatch"] else _v, 2) for _k, _v in v.items()} for k, v in json.load(f).items()}

    metric_ae = None
    if eval_file_ae is not None:
        tmp_path = pj(model_checkpoint, "eval", f"{eval_file_ae}.{dataset.replace('/', '_')}.{dataset_name}.json")
        if os.path.exists(tmp_path):
            with open(tmp_path) as f:
                metric_ae = {k: {_k: round(_v * 100 if _k not in ["AnswerF1Score", "AnswerExactMatch"] else _v, 2) for _k, _v in v.items()} for k, v in json.load(f).items()}

    metric_qag_pipeline = None
    tmp_path = pj(model_checkpoint, "eval_pipeline", f"metric.first.answer.paragraph.questions_answers.{dataset.replace('/', '_')}.{dataset_name}.{model_name.replace('-qg', '-ae').replace('/', '_')}.json")
    if os.path.exists(tmp_path):
        with open(tmp_path) as f:
            metric_qag_pipeline = {k: {_k: round(_v * 100, 2) for _k, _v in v.items()} for k, v in json.load(f).items()}
    metric_main = [dataset, dataset_name, metric, metric_qag, metric_qa, metric_ae, metric_qag_pipeline]

    # metric for ood
    metrics_ood = []
    for i in sorted(glob(pj(model_checkpoint, "eval_ood", f"{eval_file}.*.json"))):
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
            metrics_ood.append([_dataset, _dataset_name, metric, None, None, None, None])

    metrics_text = '\n'.join([
        format_metric(
            dataset=d,
            dataset_type=t,
            metric=m,
            metric_qag=m_qag,
            metric_qa=m_qa,
            metric_ae=m_ae,
            metric_qag_pipe=m_qag_pipe,
            is_multitask=_is_multitask,
            is_end2end=_is_qag,
            is_qa=_is_qa,
            is_ae=_is_ae
        ) for d, t, m, m_qag, m_qa, m_ae, m_qag_pipe in [metric_main] + metrics_ood])
    # readme table
    df_main = pd.DataFrame(*list(zip(*list(metric_main[2]["test"].items())))[::-1], columns=["Score"])
    df_main['Type'] = metric_main[1]
    df_main['Dataset'] = f"[{metric_main[0]}](https://huggingface.co/datasets/{metric_main[0]})"
    link_main = f'https://huggingface.co/{model_name}/raw/main/eval/{eval_file}.{dataset.replace("/", "_")}.{dataset_name}.json'
    df_main = df_main.sort_index()
    if metric_title == "Question & Answer Generation":
        df_main = keep_qag_metric(df_main)
    markdown_table = f"""
- ***Metric ({metric_title})***: [raw metric file]({link_main}) 

{df_main.to_markdown()}

"""
    if metric_main[3] is not None:
        df_qag = pd.DataFrame(*list(zip(*list(metric_main[3]["test"].items())))[::-1], columns=["Score"])
        df_qag['Type'] = metric_main[1]
        df_qag['Dataset'] = f"[{metric_main[0]}](https://huggingface.co/datasets/{metric_main[0]})"
        link_qag = f'https://huggingface.co/{model_name}/raw/main/eval/{eval_file_qag}.{dataset.replace("/", "_")}.{dataset_name}.json'
        df_qag = df_qag.sort_index()
        df_qag = keep_qag_metric(df_qag)
        markdown_table += f"""
- ***Metric (Question & Answer Generation{', Reference Answer' if not _is_multitask else ''})***: {"" if _is_multitask else "Each question is generated from *the gold answer*."} [raw metric file]({link_qag})

{df_qag.to_markdown()}

"""
    if metric_main[6] is not None:
        df_qag_pipe = pd.DataFrame(*list(zip(*list(metric_main[6]["test"].items())))[::-1], columns=["Score"])
        df_qag_pipe['Type'] = metric_main[1]
        df_qag_pipe['Dataset'] = f"[{metric_main[0]}](https://huggingface.co/datasets/{metric_main[0]})"
        link_qag_pipe = f"https://huggingface.co/{model_name}/raw/main/eval_pipeline/metric.first.answer.paragraph.questions_answers.{dataset.replace('/', '_')}.{dataset_name}.{model_name.replace('-qg', '-ae').replace('/', '_')}.json"
        ae_model = f"https://huggingface.co/{model_name.replace('-qg', '-ae')}"
        df_qag_pipe = df_qag_pipe.sort_index()
        df_qag_pipe = keep_qag_metric(df_qag_pipe)
        markdown_table += f"""
- ***Metric (Question & Answer Generation, Pipeline Approach)***: Each question is generated on the answer generated by [`{model_name.replace('-qg', '-ae')}`]({ae_model}). [raw metric file]({link_qag_pipe})

{df_qag_pipe.to_markdown()}

"""
    if metric_main[4] is not None:
        df_qa = pd.DataFrame(*list(zip(*list(metric_main[4]["test"].items())))[::-1], columns=["Score"])
        df_qa['Type'] = metric_main[1]
        df_qa['Dataset'] = f"[{metric_main[0]}](https://huggingface.co/datasets/{metric_main[0]})"
        link_qa = f'https://huggingface.co/{model_name}/raw/main/eval/{eval_file_qa}.{dataset.replace("/", "_")}.{dataset_name}.json'
        df_qa = df_qa.sort_index()
        markdown_table += f"""
- ***Metric (Question Answering)***: [raw metric file]({link_qa})

{df_qa.to_markdown()}

"""
    if metric_main[5] is not None:
        df_ae = pd.DataFrame(*list(zip(*list(metric_main[5]["test"].items())))[::-1], columns=["Score"])
        df_ae['Type'] = metric_main[1]
        df_ae['Dataset'] = f"[{metric_main[0]}](https://huggingface.co/datasets/{metric_main[0]})"
        link_ae = f'https://huggingface.co/{model_name}/raw/main/eval/{eval_file_ae}.{dataset.replace("/", "_")}.{dataset_name}.json'
        df_ae = df_ae.sort_index()
        markdown_table += f"""
- ***Metric (Answer Extraction)***: [raw metric file]({link_ae})

{df_ae.to_markdown()}

"""
    if len(metrics_ood) != 0:
        content = "\n".join([
                      f"| [{d}](https://huggingface.co/datasets/{d}) | {t} | {round(100 * m['test']['BERTScore'], 2)} | {round(100 * m['test']['Bleu_4'], 2)} | {round(100 * m['test']['METEOR'], 2)} | {round(100 * m['test']['MoverScore'], 2)} | {round(100 * m['test']['ROUGE_L'], 2)} | "
                      f"[link](https://huggingface.co/{model_name}/raw/main/eval_ood/{eval_file}.{d.replace('/', '_')}.{t}.json) |"
                      for d, t, m, _, _, _, _ in metrics_ood])
        markdown_table += f"""
- ***Metrics ({metric_title}, Out-of-Domain)***
        
| Dataset | Type | BERTScore| Bleu_4 | METEOR | MoverScore | ROUGE_L | Link |
|:--------|:-----|---------:|-------:|-------:|-----------:|--------:|-----:|
{content}
"""
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
{metrics_text}
---

# Model Card of `{model_name}`
{header}
{add_info}

### Overview
- **Language model:** [{language_model}](https://huggingface.co/{language_model})   
- **Language:** {la}  
- **Training data:** [{dataset}](https://huggingface.co/datasets/{dataset}) ({dataset_name})
- **Online Demo:** [https://autoqg.net/](https://autoqg.net/)
- **Repository:** [https://github.com/asahi417/lm-question-generation](https://github.com/asahi417/lm-question-generation)
- **Paper:** {paper_link}

### Usage
- With [`lmqg`](https://github.com/asahi417/lm-question-generation#lmqg-language-model-for-question-generation-)
```python
{usage_lmqg}
```

- With `transformers`
```python
{usage}
```

## Evaluation

{markdown_table}

## Training hyperparameters

The following hyperparameters were used during fine-tuning:
{config_text}

The full configuration can be found at [fine-tuning config file](https://huggingface.co/{model_name}/raw/main/trainer_config.json).

## Citation
```
{bib}
```
"""
