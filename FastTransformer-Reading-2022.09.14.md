## FastTransformer-Reading-2022.09.14

### TODO

1. Try to run the fast transformer with a correct example
2. figure out how does python transfer weights to c++ weights how to orgranize it
3. find calling chain do we call these four GEMM functions 
4. try to modify weights pointer \& print weight to the GEMM function \& find the weights if the other part of code use \& wait to add a gemm method

### Run

1. cmake .
2. make

### parameter passing

1. gpt_example.py: init GPT class (defined in utils gpt.py) & call GPT.load
2. gpt.py used GPTWeights to load weights into self.w (a list) weights.w. So there is a self.weights.w.
Pay attention to that he load the weights from 
3. gpt.py build this torch.classes.load_library by self.model = torch.classes.FasterTransformer.GptOp
4. So when we call gpt it will cal forward => calling self.model.forward
5. model = gptop.cc GptOp use ftgpt->forward & std::vector<th::Tensor> weights of gptop.cc
Q: In general class that used in pytorch will insert customclass, why fasttransformer class do not inherit TORCH_LIBRARY(my_classes, m) in the custom_class.h?
It does. It inherit from GptOp:: th::jit::CustomClassHolder - namespace th = torch; So it still use pytorch customclassholder.

6. ftgpt forward [init a cublas and then use ft == ft::ParallelGpt->forward(,,gpt_weights). Also modify input_tensors (hashmap)] & init gpt_weights (ft::ParallelGptWeight) by weights

7. ParallelGpt : class ParallelGpt in it's forward process we have  
    1. gpt_context_decoder_->forward(&decoder_output_tensors, &decoder_input_tensors, &gpt_weights->decoder_layer_weights);
    during the forward, it's unordered_map, unordered_map the 526 line 's overload function. gpt_context_decoder_ it's a ParallelGptContextDecoder and it have a const std::vector<ParallelGptDecoderLayerWeight<T>*>* gpt_decoder_layer_weight
        

        1. self_attention_layer_ (TensorParallelGptContextAttentionLayer)->forward
            1. call GptContextAttentionLayer<T>::forward and at first it will call GptContextAttentionLayer<T>::forward
                Q,K,V：
                cublas_wrapper_->Gemm(CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              3 * local_hidden_units_,  // n
                              m,
                              hidden_units_,  // k
                              attention_weights->query_weight.kernel,
                              3 * local_hidden_units_,  // n
                              attention_input,
                              hidden_units_,  // k
                              qkv_buf_,
                              3 * local_hidden_units_ /* n */);

                output:
                cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  hidden_units_,
                                  m,
                                  local_hidden_units_,
                                  attention_weights->attention_output_weight.kernel,
                                  hidden_units_,
                                  qkv_buf_3_,
                                  local_hidden_units_,
                                  attention_out,
                                  hidden_units_);

        2. ffn_layer_->forward (TensorParallelGeluFfnLayer)->forwad
            it will call TensorParallelGeluFfnLayer forward
            then it will call Ffn Layer
            
            Intermidate: cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  inter_size_,
                                  m,
                                  hidden_units_,
                                  ffn_weights->intermediate_weight.kernel,
                                  inter_size_,
                                  input_tensor,
                                  hidden_units_,
                                  inter_buf_,
                                  inter_size_);

            cublas_wrapper_->Gemm(CUBLAS_OP_N,
                                  CUBLAS_OP_N,
                                  hidden_units_,
                                  m,
                                  inter_size_,
                                  ffn_weights->output_weight.kernel,
                                  hidden_units_,
                                  inter_buf_,
                                  inter_size_,
                                  output_tensor,
                                  hidden_units_);

    2. gpt_decoder_

    3. dynamic_decode_layer_

## Design

Implement a cublas_wrapper_->SparseGemm

1. Replace cublas_wrapper_->Gemm above with cublas_wrapper_->SparseGemm (All of them related above 4*2 Gemm)

2. For the attention_weights->query_weight.kernel, attention_weights->attention_output_weight.kernel, ffn_weights->intermediate_weight.kernel, ffn_weights->output_weight.kernel. We need to pass by the other pointer format.
a. load



## Unit Test 

Run it print the output tensor to if it's the same

## Summarize

gpt_exampel => gpt.py (GPT [GPT Weights, model == torch.classes.FasterTransformer.GptOp]) => gptop.cc ftgpt => parallel GPT {gpt_context_decoder_, gpt_decoder_, dynamic_decode_layer_}
1. gpt_context_decoder
    1. context attention => GptContextAttentionLayer<T>::forward
    2. ffn_layer => Ffn Layer::forward
2. gpt_decoder