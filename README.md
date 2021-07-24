<b> Select a random pair of senetences from the data we prepared </b>

sample = random.choice(pairs)
sample

=>['vous etes plus intelligent que moi .', 'you re smarter than me .']   

<b> In order to work with embedding layer and the LSTM the inputs should be in the form of tensor, So we need to convert the sentences(words) to tensors. </b>
<b> First we'll split the sentences by whitespaces and convert each words into indices(using word2index[word]) </b>

input_sentence = sample[0]
target_sentence = sample[1]
input_indices = [input_lang.word2index[word] for word in input_sentence.split(' ')]
target_indices = [output_lang.word2index[word] for word in target_sentence.split(' ')]
input_indices, target_indices

=>([118, 214, 152, 135, 902, 42, 5], [129, 78, 1319, 1166, 343, 4])   

<b> Then convert the input_indices into tensors </b>

input_tensor = torch.tensor(input_indices, dtype=torch.long, device= device)
output_tensor = torch.tensor(target_indices, dtype=torch.long, device= device)
Next, We will define a Embedding layer as well as LSTM layers for encoder

embedding = nn.Embedding(input_size, hidden_size).to(device)
lstm = nn.LSTM(hidden_size, hidden_size).to(device)

<b> We are working with 1 sample, but we would be working for a batch. Let's fix that by converting our input_tensor into a fake batch </b>

print(embedded_input.shape)
embedded_input = embedding(input_tensor[0].view(-1, 1))
print(embedded_input.shape)
Let's build our LSTM, initialize the hidden state and cell state with Zeros(Empty state)

(hidden,ct) = torch.zeros(1, 1, 256, device=device),torch.zeros(1, 1, 256, device=device)
embedded_input = embedding(input_tensor[0].view(-1, 1))
output, (hidden,ct) = lstm(embedded_input, (hidden,ct))

<b> Now we will define a empty tensor with size MAX_LENGTH to store the Encoder outputs. </b>
<b> Then we can get the encoder outputs for each of the word in the Sentence </b>

encoder_outputs = torch.zeros(MAX_LENGTH, 256, device=device)
(encoder_hidden,encoder_ct) = torch.zeros(1, 1, 256, device=device),torch.zeros(1, 1, 256, device=device)

for i in range(input_tensor.size()[0]):  
  embedded_input = embedding(input_tensor[i].view(-1, 1))
  output, (encoder_hidden,encoder_ct) = lstm(embedded_input, (encoder_hidden,encoder_ct))
  encoder_outputs[i] += output[0,0]
Encoder Steps
Input Sentence: vous etes plus intelligent que moi .
Target Sentence: you re smarter than me .
Input indices: [118, 214, 152, 135, 902, 42, 5]
Target indices: [118, 214, 152, 135, 902, 42, 5]
After adding the <EOS> token
Input indices: [118, 214, 152, 135, 902, 42, 5, 1]
Target indices: [118, 214, 152, 135, 902, 42, 5, 1]
Input tensor: tensor([118, 214, 152, 135, 902,  42,   5,   1], device='cuda:0')
Target tensor: tensor([ 129,   78, 1319, 1166,  343,    4,    1], device='cuda:0')

Step 0
Word => vous
Input Tensor => tensor(118, device='cuda:0')
08 09

 Step 1
 Word => etes
 Input Tensor => tensor(214, device='cuda:0')
10 11

Step 2
Word => plus
Input Tensor => tensor(152, device='cuda:0')
12 13

Step 3
Word => intelligent
Input Tensor => tensor(135, device='cuda:0')
14 15

Step 4
Word => que
Input Tensor => tensor(902, device='cuda:0')
16 17

Step 5
Word => moi
Input Tensor => tensor(42, device='cuda:0')
18 19

Step 6
Word => .
Input Tensor => tensor(5, device='cuda:0')
20 21

Step 7
Word => <EOS>
Input Tensor => tensor(1, device='cuda:0')
22 23

<b> We completed the Encoder part now, Now we can start building the Attention Decoder </b>
<b> First input to the decoder will be SOS_token, later inputs would be the words it predicted (unless we implement teacher forcing). </b>

<b> Decoder/LSTM's hidden state will be initialized with the encoder's last hidden state.

We will use LSTM's hidden state and last prediction to generate attention weight using a FC layer.

This attention weight will be used to weigh the encoder_outputs using batch matric multiplication. This will give us a NEW view on how to look at encoder_states.

this attention applied encoder_states will then be concatenated with the input, and then sent a linear layer and then sent to the LSTM. </b>

  <b> LSTM's output will be sent to a FC layer to predict one of the output_language words </b>

first input
decoder_input = torch.tensor([[SOS_token]], device=device)
(decoder_hidden,decoder_ct) = (encoder_hidden,encoder_ct)
decoded_words = []
The inputs to LSTM are last prediction or a word from the target sentence based on teacher forcing ratio and decoder hidden states.

We need to concatenate the embeddings and the last decoder hidden state

 torch.cat((embedded[0], decoder_hidden[0]), 1).shape
 
 => torch.Size([1, 512])
Now we will calaculate the attentions. We will calculating the attentions by conacatinating the embeddings and last decoder hidden state and giving as input to the fully connected layer.

attn_weight_layer = nn.Linear(256 * 2, 10).to(device) \n
attn_weights = attn_weight_layer(torch.cat((embedded[0], decoder_hidden[0]), 1)) \n
attn_weights

=>tensor([[-0.8181,  0.0128,  0.0196, -0.3952, -0.1043, -0.1855, -0.5074, -0.4552,
     -0.5731,  0.5895]], device='cuda:0', grad_fn=<AddmmBackward>)
  
<b> Now We got the attention weights, now we'll apply the attention on the encoder outputs and combine the applied attentions and embeddings, pass through the relu, this will be input for the LSTM layer. On the output of the LSTM we'll the softmax and predict the expected word. </b>

Decoder steps with Full teacher forcing
Step 0
Expected output(word) => you 
Expected output(Index) => 129 
Predicted output(word) => opposed 
Predicted output(Index) => 2669 
24

Step 1
Expected output(word) => re 
Expected output(Index) => 78 
Predicted output(word) => opposed 
Predicted output(Index) => 2669 
25

Step 2
Expected output(word) => smarter 
Expected output(Index) => 1319 
Predicted output(word) => opposed 
Predicted output(Index) => 2669 
26

Step 3
Expected output(word) => than 
Expected output(Index) => 1166 
Predicted output(word) => options 
Predicted output(Index) => 1343 
27

Step 4
Expected output(word) => me 
Expected output(Index) => 343 
Predicted output(word) => articulate 
Predicted output(Index) => 964 
28

Step 5
Expected output(word) => . 
Expected output(Index) => 4 
Predicted output(word) => forgetting 
Predicted output(Index) => 2345 
29
