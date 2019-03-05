import collections
import os
from bert import modeling
from bert import optimization
from bert import tokenization
import tensorflow as tf


import pickle
flags = tf.flags

FLAGS = flags.FLAGS

root = '/home/hhl/code/my_bert/'
moder_dir = './model_dir/chinese_L-12_H-768_A-12'
data_dir = './data/NERdata'
# required parameters
flags.DEFINE_string('data_dir', data_dir, "The input data dir. Should contain the .tsv files (or other data files) for the task")
flags.DEFINE_string('task_name', 'NER', 'the name of task to train')
flags.DEFINE_string('vocab_file', moder_dir + '/vocab.txt', 'the vocabulary file that the BERT model was trained on')
flags.DEFINE_string('bert_config_file',moder_dir + '/bert_config.json', "the config json file corresponding to the pre-trained bert model.")
flags.DEFINE_string('output_dir', './chinese_ner_model', 'the output dir where the model checkpoints wile be written')

# other parameters
flags.DEFINE_string('init_checkpoint', moder_dir + '/uncased_L-12_H-768_A-12/bert_model.ckpt', 'Initial checkpoint (usually from a pre-trained BERT model).')
# Typically Uncased model is better unless you know that the cased information is importance for your task such as Name Entity Recognition or Part-Of-Speech tagging
flags.DEFINE_bool('do_lower_case', True, 'Whether to lower case the input text. Should be True for uncased models and False for cased model')
flags.DEFINE_integer('max_seq_length', 128, 'The maximum total input sequence length after WordPiece tokenization'
                                            'Sequences longer than this will be truncated, and sequences shorter than this will be padded')
flags.DEFINE_bool('do_train',True,'do train')
flags.DEFINE_bool('do_eval', True,'do eval')
flags.DEFINE_bool('do_predict', False,'do predict')
flags.DEFINE_integer('train_batch_size', 32,'total batch size for training')
flags.DEFINE_integer('eval_batch_size', 8,'total batch size for eval.')
flags.DEFINE_integer('predict_batch_size', 8,'total batch size for predict')
flags.DEFINE_float('learning_rate', 5e-5,'the initial learning rate for Adam')
flags.DEFINE_float('num_train_epochs', 3.0,'total number of training epochs to perform')
flags.DEFINE_float('warmup_proportion', 0.1, 'proportion of training to perform linear learning rate warmup for. E.g. 0.1 = 10% of training')
flags.DEFINE_integer('save_checkpoints_steps', 1000,'how often to save the model checkpoint')
flags.DEFINE_integer('iterations_per_loop', 1000 ,'how many step to make in each estimator call.')
flags.DEFINE_bool('use_tpu',False,'whether to use TPU or GPU/CPU')
tf.flags.DEFINE_string("tpu_name", None, "The Cloud TPU to use for training. This should be either the name used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.")
tf.flags.DEFINE_string( "tpu_zone", None,"[Optional] GCE zone where the Cloud TPU is located in. If not specified, we will attempt to automatically detect the GCE project from metadata.")
tf.flags.DEFINE_string( "gcp_project", None, "[Optional] Project name for the Cloud TPU-enabled project. If not specified, we will attempt to automatically detect the GCE project from metadata.")
tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_integer( "num_tpu_cores", 8, "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text = text
    self.label = label


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding='gbk') as f:
        lines = []
        words = []
        labels = []
        for line in f:
            contend = line.strip()
            word = contend.split(" ")[0]
            label = contend.split(" ")[-1]
            if line.startswith("-DOCSTART-"):
                words.append("")
                continue
            if len(contend) == 0 or words[-1]=='.':
               l = ' '.join([label for label in labels if len(label)> 0 ])
               w = ' '.join([word for word in words if len(word)> 0 ])
               lines.append([l, w])
               words = []
               labels = []
            words.append(word)
            labels.append(label)
        return lines

class NerProcessor(DataProcessor):
    """Processor for the NER data set."""
    
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")
    
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.txt")), "dev")
    
    def get_labels(self):
        """See base class."""
        return ["B-MISC", "I-MISC", "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
    
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(
                InputExample(guid=guid, text=text, label=label))
        
        return examples
    
def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    output_layer = model.get_sequence_output()

    hidden_size = output_layer.shape[-1].value

    output_weight = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer()
    )
    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        output_layer = tf.reshape(output_layer, [-1, hidden_size])
        logits = tf.matmul(output_layer, output_weight, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, 13])
        # mask = tf.cast(input_mask,tf.float32)
        # loss = tf.contrib.seq2seq.sequence_loss(logits,labels,mask)
        # return (loss, logits, predict)
        ##########################################################################
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_sum(per_example_loss)
        probabilities = tf.nn.softmax(logits, axis=-1)
        predict = tf.argmax(probabilities,axis=-1)
        return (loss, per_example_loss, logits,predict)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids,):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        #self.label_mask = label_mask

def convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer, mode):
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    with open('./data/ner/label2id.pkl','wb') as w:
        pickle.dump(label_map, w)
    text_list = example.text.split(' ')
    label_list = example.label.split(' ')
    tokens = []
    labels = []
    for i, word in enumerate(text_list):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = label_list[i]
        for m in range(len(token)):
            if m == 0:
                labels.append(label_1)
            else:
                labels.append("X")
    # tokens = tokenizer.tokenize(example.text)
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    #label_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        #label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    #assert len(label_mask) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        #tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        #label_mask = label_mask
    )
    # write_tokens(ntokens,mode)
    return feature

def filed_based_convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_file,mode=None):
    
    writer = tf.python_io.TFRecordWriter(output_file)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, label_list, max_seq_length, tokenizer,mode)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        #features["label_mask"] = create_int_feature(feature.label_mask)
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()
    
def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    # 1、读入并处理数据的processor
    processors = {
        'ner':NerProcessor
    }
    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case, FLAGS.init_checkpoint)
    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))
    task_name = FLAGS.task_name.lower()
    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))
    processor = processors[task_name]()
    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir)
        num_train_steps = int(len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
        
    label_list = processor.get_labels()
    
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list) + 1,
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)
    
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)
    tf.gfile.MakeDirs(FLAGS.output_dir)
    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        filed_based_convert_examples_to_features(
            train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(train_examples))
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)
    
if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()

