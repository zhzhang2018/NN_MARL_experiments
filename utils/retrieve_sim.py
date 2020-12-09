# Script for loading all models with the given prefix, and then running simulations and storing them. 
# The user is responsible for constructing their own environment. 
import glob
import os
from .plotting import plot_test

def grab_model_names(prefix, dir='models/', postfix=['_A','_C']):
	# Grab all matching names
	fnames = []
	if len(postfix) == 0:
		fnames = glob.glob(dir+prefix+'*')
	for pf in postfix:
		fn = glob.glob(dir+prefix+'*'+pf)
		fn = [f[:-len(pf)] for f in fn]
		fnames += fn

	# Remove repeated names
	fnames = list(set(fnames))
	fnames.sort()
	return [f[len(dir):] for f in fnames]

def run_sims_for_models(fnames, agent, env, num_iteration=100, action_space=[-1,1], dir='models/', imdir='screencaps/'):
	for f in fnames:
		# print(f)
		agent.load_model('models/'+f)
		plot_test(agent, env, fnames=[f],
	        num_iteration=num_iteration, 
	        action_space=action_space, 
	        imdir=imdir,
	        debug=False)
		os.system('ffmpeg -y -pattern_type glob -i "'+imdir+f+'*.jpg" '+f+'.gif')
		print('saved gif into ', f+'.gif')


if __name__ == '__main__':
	print("Reminder: You need to generate your own (template) agent and environment by some way.")

	prefix = input("Enter your prefix here: ")