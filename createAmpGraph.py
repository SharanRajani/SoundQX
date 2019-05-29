import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale = 0.9)



def makegraph(y):
	plt.figure(figsize=(9,4))
	# plt.xkcd()
	plt.xlabel("Time Stamps(44.1k/sec)")
	plt.ylabel("Amplitude")
	plt.plot(y, c = "#548194")
	plt.savefig("./static/images/AmpVsTime.png", bbox_inches = 'tight')
