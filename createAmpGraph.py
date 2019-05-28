import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale = 1.8)



def makegraph(y):
	plt.figure(figsize=(20,10))
	# plt.xkcd()
	plt.xlabel("Time Stamps(44.1k/sec)")
	plt.ylabel("Amplitude")
	plt.plot(y, c = "#548194")
	plt.savefig("AmpVsTime")