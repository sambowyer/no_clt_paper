import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.style.use(['default'])

# Set font sizes
plt.rc('font', size=8)          # controls default text sizes
plt.rc('axes', titlesize=10)    # fontsize of the axes title
plt.rc('axes', labelsize=8)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=7)    # fontsize of the tick labels
plt.rc('ytick', labelsize=7)    # fontsize of the tick labels
plt.rc('legend', fontsize=7)    # legend fontsize
plt.rc('figure', titlesize=9)   # fontsize of the figure title

# Define the parameters for the Beta distributions
a1, b1 = 10, 10
a2, b2 = 100, 20
a3, b3 = 20, 100

# Generate x values
x = np.linspace(0, 1, 1000)

# Calculate the PDF for each distribution
y1 = beta.pdf(x, a1, b1)
y2 = beta.pdf(x, a2, b2)
y3 = beta.pdf(x, a3, b3)

# Create the subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.25, 1.5), sharey=True)

# Plot the Beta(10,10) distribution
ax1.plot(x, y1, label='Beta(10,10)')
ax1.fill_between(x, y1, color='lightblue', alpha=0.5)
# ax1.set_title('Beta(10,10) PDF')
# ax1.set_xlabel('x')
ax1.set_ylabel('Density')
ax1.legend(loc='upper left')

ax1.set_yticks([0,5,10])

# Plot the Beta(100,20) distribution
ax2.plot(x, y2, label='Beta(100,20)')
ax2.fill_between(x, y2, color='lightblue', alpha=0.5)
# ax2.set_title('Beta(100,20) PDF')
# ax2.set_xlabel('x')
# ax2.set_ylabel('Density')
ax2.legend(loc='upper left')

# Plot the Beta(20,100) distribution
ax3.plot(x, y3, label='Beta(20,100)')
ax3.fill_between(x, y3, color='lightblue', alpha=0.5)
# ax3.set_title('Beta(20,100) PDF')
# ax3.set_xlabel('x')
# ax3.set_ylabel('Density')
ax3.legend(loc='upper right')

# gridlines
ax1.grid(True, alpha=0.3)
ax2.grid(True, alpha=0.3)
ax3.grid(True, alpha=0.3)

# Save the figure
plt.tight_layout()
plt.savefig('plots/pngs/beta_pdfs.png')
plt.savefig('plots/pdfs/beta_pdfs.pdf')
plt.savefig('PLOTS_FINAL/pngs/beta_pdfs.png')
plt.savefig('PLOTS_FINAL/pdfs/beta_pdfs.pdf')
plt.show()