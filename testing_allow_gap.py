# testing allow gap
		test_img = np.empty([7,7,7], dtype=int)
		test_img.fill(100)
		test_img.itemset((3,3,3), 255)

		i = 3
		j = 3
		k = 3
		# for kk in range (-1,2):
		# 	d = k+kk
		# 	for jj in range (-1,2):
		# 		h = j+jj
		# 		for ii in range (-1,2):
		# 			w = i+ii
		# 			test_img = set_neighbour(test_img,w,h,d)

		test_img = set_neighbour(test_img,3,3,3)


		test_img.itemset((3,3,3), 255)
		print(test_img)
		writetiff3d('test/allow_gap_exp/exp.tif',test_img)
		ini_swc = fastmarching_dt_tree(test_img,bimg,[7,7,7],3,3,3,0,args.allow_gap,args.ini)



		if not args.silence:
			print('--Hierarchy Prune')

			# hierarchy_prune(ini_swc)



def set_neighbour(test_img,i,j,k):
	for kk in range (-1,2):
		d = k+kk
		for jj in range (-1,2):
			h = j+jj
			for ii in range (-1,2):
				w = i+ii
				print(w,h,d)
				test_img.itemset((w,h,d),0)
	return test_img