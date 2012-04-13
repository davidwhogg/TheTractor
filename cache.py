try:
	# python 2.7
	from collections import OrderedDict
except:
	from .ordereddict import OrderedDict

'''
This code is based on: http://code.activestate.com/recipes/498245-lru-and-lfu-cache-decorators/
By: Raymond Hettinger
License: Python Software Foundation (PSF) license.
'''
class Cache(object):
	class Entry(object):
		pass
	def __init__(self, maxsize=1000, sizeattr='size'):
		self.dict = OrderedDict()
		self.hits = 0
		self.misses = 0
		self.maxsize = maxsize
		self.sizeattr = sizeattr
	def __setitem__(self, key, val):
		sz = 0
		if hasattr(val, self.sizeattr):
			try:
				sz = int(getattr(val, self.sizeattr))
			except:
				pass
		e = Cache.Entry()
		e.val = val
		e.size = sz
		e.hits = 0
		self.dict[key] = e
	def __getitem__(self, key):
		# pop
		try:
			e = self.dict.pop(key)
		except KeyError:
			self.misses += 1
			# purge LRU item
			if len(self.dict) > self.maxsize:
				self.dict.popitem(0)
			raise
		self.hits += 1
		# reinsert (to record recent use)
		self.dict[key] = e
		if e is None:
			return e
		e.hits += 1
		return e.val
	def __len__(self):
		return len(self.dict)
	def get(self, *args):
		if len(args) == 1:
			key = args[0]
			return self.__getitem__(key)
		assert(len(args) == 2)
		key,default = args
		try:
			return self.__getitem__(key)
		except:
			return default
	def about(self):
		print 'Cache has', len(self), 'items:'
		for k,v in self.dict.items():
			if v is None:
				continue
			print '  size', v.size, 'hits', v.hits
	def printStats(self):
		print 'Cache has', len(self), 'items'
		print 'Total of', self.hits, 'cache hits and', self.misses, 'misses'
		nnone = 0
		hits = 0
		size = 0
		for k,v in self.dict.items():
			if v is None:
				nnone += 1
				continue
			hits += v.hits
			size += v.size
		print '  ', nnone, 'entries are None'
		print 'Total number of hits of cache entries:', hits
		print' Total size (pixels) of cache entries:', size
		

class NullCache(object):
	def __getitem__(self, key):
		raise KeyError
	def __setitem__(self, key, val):
		pass