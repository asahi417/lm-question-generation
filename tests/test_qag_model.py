from lmqg import TransformersQG

test_en = "CBS broadcast Super Bowl 50 in the U.S., and charged an average of $5 million for a 30-second commercial during the game. The Super Bowl 50 halftime show was headlined by the British rock group Coldplay with special guest performers Beyoncé and Bruno Mars, who headlined the Super Bowl XLVII and Super Bowl XLVIII halftime shows, respectively. It was the third-most watched U.S. broadcast ever."
model_en = TransformersQG(language="en")
model_en.eval()
print(model_en.generate_a(test_en))
print(model_en.generate_qa(test_en))

test_ja = "織田信長は、日本の戦国時代から安土桃山時代にかけての武将、戦国大名。三英傑の一人。尾張国出身。織田信秀の嫡男。家督争いの混乱を収めた後に、桶狭間の戦いで今川義元を討ち取り、勢力を拡大した。足利義昭を奉じて上洛し、後には義昭を追放することで、畿内を中心に独自の中央政権を確立して天下人となった。しかし天正10年6月2日、家臣・明智光秀に謀反を起こされ、本能寺で自害した。"
model_ja = TransformersQG(language="ja")
model_ja.eval()
print(model_ja.generate_a(test_ja))
print(model_ja.generate_qa(test_ja))
