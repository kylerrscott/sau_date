
# input: a tensor
# output: a tensor, result after quantizing values in input into 2^bits values.
def kbit_quantize(input):
  bits = 1
  t = 1/(2**(bits-1))
  thresh = []
  for i in range(1-2**(bits-1), 2**(bits-1)):
    thresh.append(t*i)
  print(thresh)

  quantizations = []
  for i in range(0-2**(bits-1), 0):
    quantizations.append(t*i)
  for i in range(1, 2**(bits-1)+1):
    quantizations.append(t*i)
  print(quantizations)
  masks = [tf.math.less(input, thresh[0])]
  for i in range(1, len(thresh)):
    masks.append(tf.math.greater_equal(input, thresh[i-1]) & tf.math.less(input, thresh[i]))
  masks.append(tf.math.greater_equal(input, thresh[len(thresh)-1]))

  output = input
  for i in range(0, len(masks)):
    output = tf.where(masks[i], tf.ones_like(input) * quantizations[i], output)

  return output

# input: a tensor
# output: a tensor, result after quantizing values in input into 2^bits values, plus zero.
def kbit_quantize_with_zero(input):
  bits = 4
  t = 1/(2**bits)
  thresh = []
  for i in range(1-2**bits, 2**bits, 2):
    thresh.append(t*i)

  quantizations = []
  q = 1/(2**(bits-1))
  for i in range((2**(bits-1))*-1, 2**(bits-1)+1):
    quantizations.append(q*i)

  masks = [tf.math.less(input, thresh[0])]
  for i in range(1, len(thresh)):
    masks.append(tf.math.greater_equal(input, thresh[i-1]) & tf.math.less(input, thresh[i]))
  masks.append(tf.math.greater_equal(input, thresh[len(thresh)-1]))

  output = input
  for i in range(0, len(masks)):
    output = tf.where(masks[i], tf.ones_like(input) * quantizations[i], output)

  return output


import numpy
#introduces activation errors based on bin error rate
def input_quantize_bins(input):
  num_bins = 36
  bin_width = 2.0/num_bins
  bin_error_rates = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.065, 0, 0, 0.023, 0.038, 0.069, 0.125, 0.25, 0.366, 0.366, 0.25, 0.125, 0.069, 0.038, 0.023, 0, 0, 0.065, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  if not len(bin_error_rates) == num_bins:
    print("ERROR LEN")

  #create thresholds for each bin
  thresh = []
  for i in numpy.arange(-1*(num_bins/2-1) * bin_width, (num_bins/2-1) * bin_width+bin_width, bin_width):
    thresh.append(i)

  print(thresh)
  print(len(thresh))
  masks = [tf.math.less(input, thresh[0])]
  for i in range(1, len(thresh)):
    masks.append(tf.math.greater_equal(input, thresh[i-1]) & tf.math.less(input, thresh[i]))
  masks.append(tf.math.greater_equal(input, thresh[len(thresh)-1]))
  print("{} masks".format(len(masks)))

  output = input
  print("nb {}".format(int(num_bins/2)))
  for i, bin_mask in enumerate(masks[:int(num_bins/2)]):
    error_mask = tf.random.stateless_binomial(shape=input_shape, seed=[i, i], counts=tf.ones_like(input), probs=[bin_error_rates[i]])
    error_mask = tf.math.equal(error_mask, 1)
    #print(i)
    #print(error_mask)
    output = tf.where(bin_mask & error_mask, tf.ones_like(input), output)

  print('len')
  print(len(masks[int(num_bins/2):]))
  for i, bin_mask in enumerate(masks[int(num_bins/2):]):
    index = i + int(num_bins/2)
    error_mask = tf.random.stateless_binomial(shape=input_shape, seed=[index, index], counts=tf.ones_like(input), probs=[bin_error_rates[index]])
    error_mask = tf.math.equal(error_mask, 1)
    #print(index)
    #print(error_mask)
    output = tf.where(bin_mask & error_mask, tf.ones_like(input)*-1, output)

  output = ksign(output)
  return output

