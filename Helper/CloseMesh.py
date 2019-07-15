import os
import glob
import time
import sys
import numpy as np
import imageio
import scipy
from collections import defaultdict

MESH_PATH = 'mesh_semantic.ply'




def ParseMesh() :
	f = open (MESH_PATH, 'r')
	vertexPos = []
	faces = []
	G = defaultdict(set)
	header = True
	countV = 0
	countF = 0
	for line in f : 
		words = line.split(' ')
		if words[0].strip() == 'ply':
			continue
		if words[0].strip() == 'end_header' :
			header = False
			continue
		if header : 
			if words[1] == 'vertex' : 
				numVertices = int(words[2].strip())
			elif words[1] == 'face' :
				numFaces = int(words[2].strip())
		else :
			if countV < numVertices : 
				vertexPos.append([float(words[0].strip()), float(words[1].strip()), float(words[2].strip())])
				countV += 1
			else : 
				faces.append([int(words[1].strip()), int(words[2].strip()), int(words[3].strip())])
				G[int(words[1].strip())].add(int(words[2].strip()))
				G[int(words[1].strip())].add(int(words[3].strip()))
				G[int(words[2].strip())].add(int(words[1].strip()))
				G[int(words[2].strip())].add(int(words[3].strip()))
				G[int(words[3].strip())].add(int(words[1].strip()))
				G[int(words[3].strip())].add(int(words[2].strip()))

	faces = np.array(faces,dtype=np.int)
	vertexPos = np.array(vertexPos)
	f.close()
	return vertexPos, faces, G


def GetLowestVertices(vertexPos, faces, G) : 
	# Follow a greedy strategy.. get the lowest vertex and then traverse its neighbour vertices with the lowest z.

	#get min Z vertex.. if multiple, pic any
	minIdx = vertexPos[:,2].argmin()
	stIndex = minIdx
	boundaryVert = []
	boundaryVert.append(minIdx)
	flag = True
	curIdx = minIdx
	count = 0
	while flag :
		flag = False
		neigh = G[curIdx]
		minIdx = list(neigh)[0]
		minZ = vertexPos[minIdx][2]
		print boundaryVert
		loop = True
		for v in neigh : 
			# Iterate over all neighbours and find the one with the smallest z value! %%%%HACK%%%%
			if vertexPos[v][2] < minZ and v not in boundaryVert:
				minZ = vertexPos[v][2]
				minIdx = v
				loop = False
		if minIdx !=  stIndex and not loop: 
			boundaryVert.append(minIdx)
			curIdx = minIdx
			flag = True
		#print count
		count += 1
	return boundaryVert

def CompleteMesh(vertexPos, faces, boundaryVert) :
	# Calculate the average vertex using boundary vertices and then join all the others to it!

	for v in boundaryVert : 
		meanX = vertexPos[v][0]
		meanY = vertexPos[v][1]
		meanZ = vertexPos[v][2]

	meanX = meanX / (1.*len(boundaryVert))
	meanY = meanY / (1.*len(boundaryVert))
	meanZ = meanZ / (1.*len(boundaryVert))

	vertexPos = np.vstack((vertexPos, np.array([meanX, meanY, meanZ])))
	addedIdx = len(vertexPos)-1

	# Now since vertices were inserted in order. I believe simply connecting would give use the faces : 
	for idx, v in enumerate(boundaryVert) :
		try : 
			faces = np.vstack((faces, np.array([v, addedIdx, boundaryVert[idx+1]])))
		except :
			pass
	return vertexPos, faces

def WriteMesh(vertexPos, faces) :
	f = open('OutputMesh.ply', 'w')
	# Write header
	f.write('ply\nformat ascii 1.0\nelement vertex {0}\nproperty float x\nproperty float y\nproperty float z\nelement face {1}\nproperty list uchar int vertex_index\nend_header\n'.format(len(vertexPos), len(faces))) 

	#Write vertices
	f.close()
	f = open('OutputMesh.ply', 'a')
	np.savetxt(f, vertexPos, fmt='%6f')
	faces = np.hstack((3*np.ones((len(faces), 1),dtype=np.int),faces))
	np.savetxt(f, faces, fmt='%i')

def main() :
	v, f, g = ParseMesh()
	b = GetLowestVertices(v, f, g)
	print b
	v, f = CompleteMesh(v, f, b)
	WriteMesh(v, f)

if __name__ == '__main__':
	main()

