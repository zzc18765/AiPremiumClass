import fasttext

model = fasttext.train_supervised('week5/cooking.stackexchange.txt', epoch=10, dim=300)
# 文本分类功能
print(model.predict("Dangerous pathogens capable of growing in acidic environments"))
print(model.predict("Baking chicken in oven, but keeping it moist"))