##############################################################
# Integrate PBay Spectrum with actualy Alluxa profile to get Measurements
# To make spectrum, go to Kavli research folder where PBay is stored
# Outputs: Plots saved to plots folder
# Inputs: Spectra from pbay + their log should be saved to spectra folder
###############################################################

import numpy as np
import matplotlib.pylab as plt
from scipy.interpolate import interp1d
from scipy.integrate import trapz
import matplotlib,sys,os
plt.ion()
from astropy.io import fits
from joblib import Parallel, delayed
import multiprocessing


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)

sys.path.append('./tools/')
from functions import *

from objects import load_object
from load_inputs import load_inputs
so = load_object('calc_signal.cfg') # load storage object and load config file
lin = load_inputs()
so = lin.load_all(so) # must do this if want different star type

#if len(sys.argv) > 1:
#	so.run.ispec = int(sys.argv[1]) - 1  # subtract 1 bc 0 isnt valid job id


######################### LOAD THINGS #####################

######################### Calc Signal Fxn #################

class calc_flux():
	"""
	calc flux using phoenix spectrum...reload spectrum for each type before running
	then choose magnitude
	"""
	def __init__(self,so,setup='alluxa',radlow=6371,specrad0=0.2):
		#define things?
		
		iflatten    = np.where(np.sqrt(so.exo.s*(specrad0*so.const.rsun)**2) < radlow)[0]
		ytemp       = 1.0 * so.exo.s * (specrad0/so.stel.rad)**2
		ytemp[iflatten] = radlow**2/(so.stel.rad*so.const.rsun)**2
		transmission    = 1-ytemp
	
		self.source     = so.stel.s * 10**(-0.4*so.stel.I_mag) * transmission
		self.source_out = so.stel.s * 10**(-0.4*so.stel.I_mag) 
		
		# Pick bandpass
		if setup=='alluxa':
			self.onband = so.filt.s_alluxa_on
			self.offband  = so.filt.s_alluxa_off
		elif setup=='alluxa_wide':
			self.onband = so.filt.s_alluxa_wide_on
			self.offband  = so.filt.s_alluxa_wide_off
	
		# Shift source to user specified velocity (default is 0.0)
		if so.stel.vel != 0.0:
			x_temp = so.exo.v + so.exo.v*(1.0*so.stel.vel/300000.0)
			# shift stellar
			interpx_shift      = interp1d(x_temp,self.source,bounds_error=False,fill_value=0)
			interpx_shift_out  = interp1d(x_temp,self.source_out,bounds_error=False,fill_value=0)
			self.source           = interpx_shift(so.exo.v)
			self.source_out       = interpx_shift_out(so.exo.v)
			# Shift dualons
			interpx_onband   = interp1d(x_temp,self.onband,bounds_error=False,fill_value=0)
			interpx_offband  = interp1d(x_temp,self.offband,bounds_error=False,fill_value=0)
			self.onband  = interpx_onband(so.exo.v)
			self.offband = interpx_offband(so.exo.v)
		
		# multiply by telluric
		at_scope     = self.source     * so.inst.tel_area #* so.tel.o2 * so.tel.rayleigh 
		at_scope_out = self.source_out * so.inst.tel_area #* so.tel.o2 * so.tel.rayleigh

		# Instrument
		at_ccd       = at_scope     * so.inst.exp_time * so.inst.qe * so.inst.tel_reflectivity
		at_ccd_out   = at_scope_out * so.inst.exp_time * so.inst.qe * so.inst.tel_reflectivity
	
		# Integrate
		self.onsignal   = integrate(so.exo.v,self.onband*at_ccd)
		self.offsignal  = integrate(so.exo.v,self.offband*at_ccd)
		self.onsignal_out   = integrate(so.exo.v,self.onband*at_ccd_out)
		self.offsignal_out  = integrate(so.exo.v,self.offband*at_ccd_out)

		signal_ppm = (self.onsignal_out/self.offsignal_out - self.onsignal/self.offsignal)*1e6
		
		self.signal_ppm = signal_ppm
		
	
	def calc_noise_one(self,so,S,setup='alluxa'):
		# Read Noise
		R  = float(so.inst.read) * so.inst.npix   # variance in electrons per pixel, bin by 4
		D  = float(so.inst.dark) * so.inst.exp_time * so.inst.npix # variance in electrons per pixel
		if setup=='alluxa':
			B  = so.inst.skybkg_photons_alluxa
		elif setup == 'alluxa_wide':
			B = so.inst.skybkg_photons_alluxa_wide
#		F  = self.flat
		
		# no scintillation
		noise_sq_1 = S  + (1+so.inst.npix/so.inst.nb)*(B + D + R) # + (F*S)**2  

		return np.sqrt(noise_sq_1)

	def calc_noise_tot(self):
		"""
		calculate the total noise
		"""
		noise_on      = self.calc_noise_one(so,self.onsignal)
		noise_off     = self.calc_noise_one(so,self.offsignal)
		noise_on_out  = self.calc_noise_one(so,self.onsignal_out)
		noise_off_out = self.calc_noise_one(so,self.offsignal_out)
		
		# Propogate noise
		frat_out = self.onsignal_out/self.offsignal_out
		frat_in  = self.onsignal/self.offsignal
	  
		noise_in  = frat_in  * np.sqrt(noise_on**2/self.onsignal**2 + noise_off**2/self.offsignal**2)
		noise_out = frat_out * np.sqrt(noise_on_out**2/self.onsignal_out**2 + noise_off_out**2/self.offsignal_out**2)
			
		self.noise_ppm = 1e6 * np.sqrt(noise_out**2 + noise_in**2)
		
		# Signal to noise ratio
		self.snr       = self.signal_ppm/self.noise_ppm
		self.nexp      = (3.0/self.snr)**2
		self.ntransits = (self.nexp * so.inst.exp_time)/so.stel.transit_duration*3600.0 # units of both in seconds
		
		return self.noise_ppm


	def plot_setup(self):
		# plot
		fig, (ax, ax3) = plt.subplots(2, 1, sharex=True,figsize=(10,6))

		ax.plot(so.exo.v,self.source/self.source_out)

		ax2 = ax.twinx() # use this for stellar contamination later
		ax2.set_zorder(ax.get_zorder()+1) # put ax in front of ax2 
		ax2.fill_between(so.exo.v,y1=self.onband,y2=0*so.exo.v,facecolor='m',alpha=0.5)
		ax2.fill_between(so.exo.v,y1=self.offband,y2=0*so.exo.v,facecolor='g',alpha=0.5)
		
		ax.set_xlim(755,770)
		ax.set_title('Type: %s  Dist: %s pc  Vel: %s'%(so.stel.type,so.stel.dist,so.stel.vel))
	
		ax2.set_ylim(0,0.9)
		ax.set_ylabel('($R_{p}/R_*)^2$')
		ax2.set_ylabel('Etalon Transmission')
	
		# Add stellar spectrum
		ax3.plot(so.exo.v,self.source_out,'k')
		ax3.set_xlabel('Wavelength (nm)')
		ax3.set_ylabel('Stellar Flux Density \n (phot/s/m$^2$/nm)')

		ax4 = ax3.twinx() # use this for stellar contamination later
		ax4.set_zorder(ax.get_zorder()+1) # put ax in front of ax2 
		ax4.set_ylabel('Sky Transmission')
	
		ax4.plot(so.exo.v,so.tel.o2)
		ax4.plot(so.exo.v,so.tel.rayleigh)
		
		fig.subplots_adjust(bottom=0.1,left=0.22,hspace=0,right=0.88,top=0.9)

		plt.savefig('./plots/filter_setup.png')



#####################################
# Loop through distances and save snr_arr 
#####################################

so = lin.reload_stellar(so,so.const.types[5],10)
diameters = np.array([ 9.6, 15.0, 21.6, 28.9, 35.3])
distances = np.array([1.8, 5.0, 8.0, 10.0, 15.0])

snr_M4 = np.zeros((len(so.const.telescopes),len(distances)))
ntransits = np.zeros((len(so.const.telescopes),len(distances)))
setup='alluxa'
if setup=='alluxa':
	fwhm = 0.3
elif setup=='alluxa_wide':
	fwhm = 2.0

for j in range(len(distances)):
	so = lin.reload_stellar(so,so.const.types[5],distances[j])
	so.stel.vel = -110.0
	for iobs, observatory in enumerate(so.const.telescopes):
		so.inst.telescope = observatory
		so = lin.instrument(so)
		cf = calc_flux(so,setup='alluxa')
		noise_tot = cf.calc_noise_tot()
		snr_M4[iobs,j] = cf.snr
		ntransits[iobs,j] = cf.ntransits
		
		
nexp_M4 = (3.0/snr_M4)**2

ploton=True
if ploton ==True:
    cmap=plt.cm.Blues
    norm = matplotlib.colors.BoundaryNorm(np.array([0,10,100,1e3,np.round(np.max(nexp_M4))]), cmap.N)
    fig, ax = plt.subplots(1,1,figsize=(9,8))
    im = ax.imshow(nexp_M4,cmap=cmap,norm=norm)
    ax.set_xticks(np.arange(len(distances)))
    ax.set_yticks(np.arange(len(so.const.telescopes)))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(distances)
    ax.set_yticklabels(so.const.telescopes)
    #cbar = fig.colorbar(im)
    #cbar.ax.set_yticklabels(['0','1','3','5','10+'])
    #cbar.ax.set_ylabel('SNR')
    plt.xlabel('Distance (pc)')
    plt.ylabel('Observatory')
    annotate_heatmap(im)
    plt.title('M4V, FWHM = %s nm' %np.round(fwhm,1))
    #plt.title('$\mathrm{Filter \: FWHM}$ = %s nm' %np.round(c.fwhm,1))
    fig.subplots_adjust(bottom=0.15,left=0.15)
    #plt.savefig('./plots/transits_snr3_fwhm_%s.pdf' %(c.fwhm))


print x

#####################################
# Calculate Flux - use vel=-100 km/s 
#####################################
so = lin.filter(so)

dist        = np.arange(1,11)
signal      = np.zeros(((len(dist)),len(so.const.types)))
noise       = np.zeros(((len(dist)),len(so.const.types)))
signal2       = np.zeros(((len(dist)),len(so.const.types)))
noise2      = np.zeros(((len(dist)),len(so.const.types)))
signal3       = np.zeros(((len(dist)),len(so.const.types)))
noise3      = np.zeros(((len(dist)),len(so.const.types)))
for i in range(len(so.const.types)):
	for j in range(len(dist)):
		so = lin.reload_stellar(so,so.const.types[i],dist[j])
		radlow = so.const.radlows[np.where(so.const.types == so.stel.type)[0]]

		# etalon
		so.stel.vel = -25 #km/s
		s, n , onsignal, offsignal, onsignal_out, offsignal_out \
				= calc_flux(so,setup='dualon_blue',radlow=radlow,plot_on=False)
		signal[j,i]  = s 
		noise[j,i]   = n 

		# Alluxa
		so.stel.vel = -125 #km/s
		s, n , onsignal, offsignal, onsignal_out, offsignal_out \
				= calc_flux(so,setup='alluxa',radlow=radlow,plot_on=False)
		signal2[j,i]   = s
		noise2[j,i]    = n

		# Alluxa Wide
		so.stel.vel = -25 #km/s
		s, n , onsignal, offsignal, onsignal_out, offsignal_out \
				= calc_flux(so,setup='alluxa_wide',radlow=radlow,plot_on=False)
		signal3[j,i]   = s
		noise3[j,i]    = n



f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True,figsize=(6,7))
for i in range(0,11):
	ax1.plot(dist,(30.0/60.0)*(3.0/(signal[:,i]/noise[:,i]))**2,c=so.const.colors[i], \
				ls=so.const.symbols[i],label=so.const.types[i])
	ax2.plot(dist,(30.0/60.0)*(3.0/(signal2[:,i]/noise2[:,i]))**2,c=so.const.colors[i], \
				ls=so.const.symbols[i],label=so.const.types[i])
	ax3.plot(dist,(30.0/60.0)*(3.0/(signal3[:,i]/noise3[:,i]))**2,c=so.const.colors[i], \
				ls=so.const.symbols[i],label=so.const.types[i])

if so.inst.telescope == 'ELT':
	ax1.set_xlim(5,10)
	ax1.text(8,0,'Dualon Red',fontsize=14)
	ax2.text(8.4,0,'Alluxa',fontsize=14)
	ax3.text(8,0,'Alluxa Wide',fontsize=14)
	ax1.text(5.15,175,'100hr < $t_{\:3\sigma}$ < 200hr',fontsize=8)
elif so.inst.telescope=='GMT':
	ax1.set_xlim(3,7)
	ax1.text(5,0,'Dualon Red',fontsize=14)
	ax2.text(5.4,0,'Alluxa',fontsize=14)
	ax3.text(5,0,'Alluxa Wide',fontsize=14)

ax1.set_ylim(-40,300)
ax2.set_ylim(-40,700)
ax3.set_ylim(-40,440)
ax1.set_title('Instrument: %s'%so.inst.telescope,fontsize=14)
ax3.set_xlabel('Distance (pc)')
ax1.set_ylabel('$t_{\:3\sigma}$ \n (hrs)',rotation=0,labelpad=20)
ax2.set_ylabel('$t_{\:3\sigma}$ \n (hrs)',rotation=0,labelpad=20)
ax3.set_ylabel('$t_{\:3\sigma}$ \n (hrs)',rotation=0,labelpad=20)
ax1.fill_between(dist,100,200,facecolor='gray',alpha=0.3)
ax2.fill_between(dist,100,200,facecolor='gray',alpha=0.3)
ax3.fill_between(dist,100,200,facecolor='gray',alpha=0.3)
f.subplots_adjust(bottom=0.1,left=0.23,hspace=0,right=0.77,top=0.95)
ax2.legend(frameon=False,fontsize=14,loc=(1.05,-0.3))

#f.savefig('./plots/hours_to_3sigma_%s.pdf' %so.inst.telescope)
