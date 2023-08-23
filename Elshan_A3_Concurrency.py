import sys
import time
import numpy as np
import cv2
from cv2 import dnn
from tkinter import *
from PIL import ImageTk, Image
import imutils
import multiprocessing as mp
import psutil 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

arggg1 = str(sys.argv[1])
arggg2 = int(sys.argv[2])
arggg3 = str(sys.argv[3])

# print(arggg1, type(arggg1))
# print(arggg2, type(arggg2))
# print(arggg3, type(arggg3))

if arggg3 == 'S':

	proto_file = 'colorization_deploy_v2.prototxt'
	model_file = 'colorization_release_v2.caffemodel'
	hull_pts = 'pts_in_hull.npy'
	net = dnn.readNetFromCaffe(proto_file,model_file)
	kernel = np.load(hull_pts)
	zzz = Image.open(arggg1).convert('RGB')
	im_array = np.array(zzz)
	
	def task():
		BBB["state"] = "disabled"
		global im_array
		global arggg2
		global arggg1
		or_h = im_array.shape[0]
		or_w = im_array.shape[1]
		if (im_array.shape[0]%arggg2)!=0:
			nexvatka = arggg2 - im_array.shape[0]%arggg2
			im_pad = np.zeros((nexvatka,im_array.shape[1],im_array.shape[2]))
			im_array = np.vstack((im_array,im_pad))

		if (im_array.shape[1]%arggg2)!=0:
			nexvatka = arggg2 - im_array.shape[1]%arggg2
			im_pad = np.zeros((im_array.shape[0],nexvatka,im_array.shape[2]))
			im_array = np.hstack((im_array,im_pad))

		for i in range(0, im_array.shape[0]-arggg2+1,arggg2):
			for j in range(0, im_array.shape[1]-arggg2+1, arggg2):
				print("i=",i," j=",j)
				img = im_array[i:i+arggg2,j:j+arggg2,:]
				scaled = img.astype("float32") / 255.0
				lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
				#-----------------------------------#---------------------#

				# add the cluster centers as 1x1 convolutions to the model
				class8 = net.getLayerId("class8_ab")
				conv8 = net.getLayerId("conv8_313_rh")
				pts = kernel.transpose().reshape(2, 313, 1, 1)
				net.getLayer(class8).blobs = [pts.astype("float32")]
				net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
				#-----------------------------------#---------------------#

				# we'll resize the image for the network
				resized = cv2.resize(lab_img, (224, 224))
				# split the L channel
				L = cv2.split(resized)[0]
				# mean subtraction
				L -= 50
				#-----------------------------------#---------------------#

				# predicting the ab channels from the input L channel

				net.setInput(cv2.dnn.blobFromImage(L))
				ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
				# resize the predicted 'ab' volume to the same dimensions as our
				# input image
				ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))


				# Take the L channel from the image
				L = cv2.split(lab_img)[0]
				# Join the L channel with predicted ab channel
				colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)

				# Then convert the image from Lab to BGR
				colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
				colorized = np.clip(colorized, 0, 1)

				# change the image to 0-255 range and convert it from float32 to int
				colorized = (255 * colorized).astype("uint8")
				im_array[i:i+arggg2,j:j+arggg2,:] = colorized
				
				demo_im = Image.fromarray(np.uint8(im_array[:or_h,:or_w,:])).convert('RGB')
				demo_im = demo_im.save("result.jpg")
				im_array1 = imutils.resize(im_array[:or_h,:or_w,:],width=500)
				demo_im1 = Image.fromarray(np.uint8(im_array1)).convert('RGB')
				demo_im1 = demo_im1.save("vremenniy_fayl.jpg")
				img_001 = ImageTk.PhotoImage(Image.open("vremenniy_fayl.jpg")) 
				canvas.delete("all")
				canvas.create_image(5, 5, anchor=NW, image=img_001) 
		canvas.create_text(100,100, text="Saved the file as result.jpg")

	root = Tk()
	root.title("Elshan Naghizade A3 - Concurrency")
	root.resizable(False, False)
	root.geometry('600x600')
	canvas = Canvas(root, width = 500, height = 500)

	im_array2 = imutils.resize(im_array,width=500)
	demo_im2 = Image.fromarray(np.uint8(im_array2)).convert('RGB')
	demo_im2 = demo_im2.save("vremenniy_fayl.jpg")
	img_002 = ImageTk.PhotoImage(Image.open("vremenniy_fayl.jpg")) 

	canvas.pack()
	canvas.create_text(100,100, text="Click the start button to track the progress")
	canvas.create_image(5, 5, anchor=NW, image=img_002) 

	import threading
	def tr_pr():
		GA_thread = threading.Thread(target=task)
		GA_thread.start()

	BBB = Button(root, text ="S T A R T", command = tr_pr)
	BBB.pack(padx=5)
	root.mainloop()

if arggg3 == 'M':
	n_cores = psutil.cpu_count(logical = False)

	proto_file = 'colorization_deploy_v2.prototxt'
	model_file = 'colorization_release_v2.caffemodel'
	hull_pts = 'pts_in_hull.npy'
	net = dnn.readNetFromCaffe(proto_file,model_file)
	kernel = np.load(hull_pts)
	zzz = Image.open(arggg1).convert('RGB')
	im_array = np.array(zzz)
	zzz.close()
	chunk_list = []
	if (im_array.shape[0]%n_cores)==0:
		for ii in range(n_cores):
			chunk_list.append(im_array[((im_array.shape[0]//n_cores)*(ii)):((im_array.shape[0]//n_cores)*(ii+1)),:,:])
	else:
		n_ost = im_array.shape[0]%n_cores
		chunk_list.append(im_array[:n_ost,:,:])
		n_cel_cores = n_cores-1
		im_array_cel = im_array[n_ost:,:,:]
		for jj in range(n_cel_cores):
			chunk_list.append(im_array_cel[((im_array_cel.shape[0]//n_cel_cores)*(jj)):((im_array_cel.shape[0]//n_cel_cores)*(jj+1)),:,:])
	# sdelay tut seriye fotki promejutochniye
	for lp in range(n_cores):
		fiiq = Image.fromarray(np.uint8(chunk_list[lp])).convert('RGB')
		fiiq = fiiq.save("ff"+str(lp)+".jpg")
		#fiiq.close()
	def task_p(input_im, arggg2,ippp):
		or_h = input_im.shape[0]
		or_w = input_im.shape[1]
		if (input_im.shape[0]%arggg2)!=0:
			nexvatka = arggg2 - input_im.shape[0]%arggg2
			im_pad = np.zeros((nexvatka,input_im.shape[1],input_im.shape[2]))
			input_im = np.vstack((input_im,im_pad))

		if (input_im.shape[1]%arggg2)!=0:
			nexvatka = arggg2 - input_im.shape[1]%arggg2
			im_pad = np.zeros((input_im.shape[0],nexvatka,input_im.shape[2]))
			input_im = np.hstack((input_im,im_pad))

		for i in range(0, input_im.shape[0]-arggg2+1,arggg2):
			for j in range(0, input_im.shape[1]-arggg2+1, arggg2):
				#print("i=",i," j=",j)
				img = input_im[i:i+arggg2,j:j+arggg2,:]
				scaled = img.astype("float32") / 255.0
				lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
				#-----------------------------------#---------------------#

				# add the cluster centers as 1x1 convolutions to the model
				class8 = net.getLayerId("class8_ab")
				conv8 = net.getLayerId("conv8_313_rh")
				pts = kernel.transpose().reshape(2, 313, 1, 1)
				net.getLayer(class8).blobs = [pts.astype("float32")]
				net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
				#-----------------------------------#---------------------#

				# we'll resize the image for the network
				resized = cv2.resize(lab_img, (224, 224))
				# split the L channel
				L = cv2.split(resized)[0]
				# mean subtraction
				L -= 50
				#-----------------------------------#---------------------#

				# predicting the ab channels from the input L channel

				net.setInput(cv2.dnn.blobFromImage(L))
				ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
				# resize the predicted 'ab' volume to the same dimensions as our
				# input image
				ab_channel = cv2.resize(ab_channel, (img.shape[1], img.shape[0]))


				# Take the L channel from the image
				L = cv2.split(lab_img)[0]
				# Join the L channel with predicted ab channel
				colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)

				# Then convert the image from Lab to BGR
				colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
				colorized = np.clip(colorized, 0, 1)

				# change the image to 0-255 range and convert it from float32 to int
				colorized = (255 * colorized).astype("uint8")
				input_im[i:i+arggg2,j:j+arggg2,:] = colorized
				ifert = input_im[:or_h,:or_w,:]
				fii = Image.fromarray(np.uint8(ifert)).convert('RGB')
				fii = fii.save("ff"+str(ippp)+".jpg")
				#fii.close()
				#fii = fii.save(str(i*j+i)+" process-> "+str(ippp)+".jpg")
	if __name__ == '__main__':
		print("Processes are getting started")
		pr_spisol = []
		for pp in range(n_cores):
			pr_spisol.append(mp.Process(target=task_p, args=(chunk_list[pp], arggg2, pp)))
		for ee in range(n_cores):
			pr_spisol[ee].start()
		print("Processes have been started !")
		
		for lll in range(50):
			print("printining "+"tempo_file_"+str(lll)+".jpg")
			time.sleep(0.5)
			ss = Image.open('ff0.jpg').convert('RGB')
			fin_res1 = np.array(ss)
			ss.close()
			for vc1 in range(1,n_cores):
				qaz = Image.open('ff'+str(vc1)+'.jpg').convert('RGB')
				tmpe1 = np.array(qaz)
				qaz.close()
				fin_res1 = np.vstack((fin_res1,tmpe1))

			fr_res1 = Image.fromarray(np.uint8(fin_res1)).convert('RGB')
			fr_res1 = fr_res1.save("tempo_file_"+str(lll)+".jpg")
			#fr_res1.close()
		print("#############")
		for eee in range(n_cores):
			pr_spisol[eee].join()
		print("Finished the conversion process")

		fin_res = np.array(Image.open('ff0.jpg').convert('RGB'))
		for vc in range(1,n_cores):
			tmpe = np.array(Image.open('ff'+str(vc)+'.jpg').convert('RGB'))
			fin_res = np.vstack((fin_res,tmpe))
		fr_res = Image.fromarray(np.uint8(fin_res)).convert('RGB')
		fr_res = fr_res.save("result.jpg")
		print("saved the final result as result.jpg")
		# canvas1.create_text(100,100, text="Saved the file as result.jpg")


		# root1.mainloop()




