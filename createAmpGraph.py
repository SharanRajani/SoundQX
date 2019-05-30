import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale = 0.9)



def makegraph(y):
	plt.figure(figsize=(8.5,3.5))
	# plt.xkcd()
	plt.xlabel("Time Stamps(44.1k/sec)")
	plt.ylabel("Amplitude")
	plt.plot(y, c = "#0b353b")
	plt.savefig("./static/images/AmpVsTime.png", bbox_inches = 'tight')

def makegraph1(y):
	plt.figure(figsize=(8.5,3.5))
	# plt.xkcd()
	plt.xlabel("Time Stamps(44.1k/sec)")
	plt.ylabel("Amplitude")
	plt.plot(y, c = "#0b353b")
	plt.savefig("./static/images/AmpVsTime1.png", bbox_inches = 'tight')
