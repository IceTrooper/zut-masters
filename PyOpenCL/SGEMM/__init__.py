import numpy as np
import pyopencl as cl

VERBOSE = False
USE_FILE = True

def find_opencl_platform(name):
	platforms = cl.get_platforms()
	print(f'Number of available platforms: {len(platforms)}')
	for p in platforms:
		p_name = p.get_info(cl.platform_info.NAME)
		if(p_name.find(name) == -1):
			# Not found
			continue
		devices = p.get_devices(cl.device_type.GPU)
		if(len(devices) > 0):
			print('Required device was found')
			print(f'Number of available devices: {len(devices)}')
			return p

	raise RuntimeError('Required device was not found on any platform!')

def ordered_numpy_array(n, m, start, step):
	return np.arange(start, start + n * m * step, step, dtype=np.float32).reshape(n, m)

def profile(ev):
	print(f'Time elapsed: {ev.profile.end - ev.profile.start} ns')

def main():
	platform = find_opencl_platform('Intel')
	contextProperties = [(cl.context_properties.PLATFORM, platform)]
	context = cl.Context(dev_type=cl.device_type.GPU, properties=contextProperties)
	devices = context.get_info(cl.context_info.DEVICES)
	device = devices[0]

	# This should be the same value as in sgemm.cl kernel code. Be aware of that!
	kDim = 1200
	nDim, mDim = 4800, 3600

	A = ordered_numpy_array(nDim, kDim, 0.00001, 0.00001)
	B = ordered_numpy_array(kDim, mDim, 0.00002, 0.00002)
	# B = np.random.rand(kDim, mDim).astype(np.float32)
	C = np.zeros((nDim, mDim), dtype=np.float32)

	if(VERBOSE):
		print(A)
		print(B)
		print(C)

	mf = cl.mem_flags
	buffer_a = cl.Buffer(context, mf.READ_ONLY, A.nbytes)
	buffer_b = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=B)
	buffer_c = cl.Buffer(context, mf.WRITE_ONLY, C.nbytes)

	command_queue = cl.CommandQueue(context, device, cl.command_queue_properties.PROFILING_ENABLE)

	cl._enqueue_write_buffer(command_queue, buffer_a, A)
	# We don't need to copy buffer_b because we used COPY_HOST_PTR flag when creating.

	# You can also pass multiline string instead of reading file
	file_handle = open('sgemm.cl', 'r')
	kernel_source = file_handle.read()
	
	program = cl.Program(context, kernel_source)

	try:
		program.build()
	except:
		pbi = cl.program_build_info
		print(program.get_build_info(device, pbi.LOG))
		print(program.get_build_info(device, pbi.OPTIONS))
		print(program.get_build_info(device, pbi.STATUS))
		raise

	# kernel = cl.Kernel(program, 'Sgemm')
	# or
	kernel = program.Sgemm

	maxComputeUnits = device.get_info(cl.device_info.MAX_COMPUTE_UNITS)
	print(f'Max compute units: {maxComputeUnits}')

	global_range = (nDim,)
	local_range = (int(nDim / maxComputeUnits),)

	event = kernel(command_queue, global_range, local_range, np.uint32(nDim), np.uint32(kDim), np.uint32(mDim), buffer_a, buffer_b, buffer_c, np.empty(kDim, dtype=np.float32))
	command_queue.finish()

	profile(event)

	cl.enqueue_copy(command_queue, C, buffer_c)

	if(VERBOSE):
		print(C)

if __name__ == '__main__':
	try:
		main()
	except cl.Error as err:
		print(f'OpenCL error: {err}')
	except BaseException as err:
		print(f'Error: {err}')
	# except:
	# 	print('Unknown exception')