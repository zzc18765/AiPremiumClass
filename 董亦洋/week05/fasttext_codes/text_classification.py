import fasttext

model = fasttext.train_supervised('cooking.stackexchange.txt', epoch=10, dim=200)


print(model.predict('How do I cover up the white spots on my cast iron stove?'))
