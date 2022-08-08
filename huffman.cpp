#include "huffman.h"
#include <cstring>
#include <queue>
#include <iostream>
#include <unordered_map>
using namespace std;

//MinHeap struct definitions

struct MinHeapNode{
    char data;
    int freq;
    struct MinHeapNode *left, *right;
    
    MinHeapNode(char data, int freq){
        left = right = NULL;
        this->data = data;
        this->freq = freq;
    }
};

struct compare{
    bool operator()(MinHeapNode *left, MinHeapNode *right){
        return (left->freq > right->freq);
    }
};

void PrintCodes(struct MinHeapNode* root, string str){
    if(!root){
        return;
    }
    //check if node data is not an internal node. If not, then print
    if(root->data != '$'){
        std::cout<<root->data<<": "<<str<<endl;
    }
    PrintCodes(root->left, str + "0");
    PrintCodes(root->right, str + "1");
}

/**
 * TODO Complete this function
 **/
int huffman_encode(const unsigned char *bufin,
						  unsigned int bufinlen,
						  unsigned char **pbufout,
						  unsigned int *pbufoutlen)
{
    std::cout << "With Unordered map:"<< endl;
    
    std::unordered_map<char, int> input_map;
    
    for(int i = 0; i < bufinlen; i++){
        input_map[bufin[i]]++;    
    }
    
    int index = 0;
    int maplen = input_map.size();
//    std::cout << "Map size: " << input_map.size() << endl;
    char map_chars[maplen]= {0};
    int char_freq[maplen] = {0};
//    std::cout << "made chars and freq arrays"<< endl;
    for(auto const &pair: input_map){
        std::cout << "{" << pair.first << ": " << pair.second << "}\n";
        map_chars[index] = pair.first;
        char_freq[index] = pair.second;
        index++;
    }
/*    //View the character and frequency arrays 
    for(int i = 0; i < maplen; i++){
        std::cout << "Unique Character: " << map_chars[i] << endl;
        std::cout << "Frequency of char: " << char_freq[i] << endl;
    }
*/
    //create minHeap structure to contain Huffman tree nodes
	struct MinHeapNode *left, *right, *top;
    std::priority_queue<MinHeapNode*, vector<MinHeapNode*>, compare> minHeap;
      
    //create a leaf/node for each character in the char array
    
    for(int i=0;i<bufinlen;i++){
        minHeap.push(new MinHeapNode(map_chars[i],char_freq[i]));
    } 
    
    //complete tree; make binary representation of each character
    
    //encode the input string using the nodes 
    
    //write the coded output to pbufout
    
    return 0;
}


/**
 * TODO Complete this function
 **/
int huffman_decode(const unsigned char *bufin,
						  unsigned int bufinlen,
						  unsigned char **pbufout,
						  unsigned int *pbufoutlen)
{
	return 0;
}
