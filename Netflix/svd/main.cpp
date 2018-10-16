//
//  main.cpp
//  Netflix
//
//  Created by Zitao Wang on 5/11/18.
//  Copyright © 2018 Zitao Wang. All rights reserved.
//

#include <iostream>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <algorithm>
#include <functional>
#include <stdio.h>
#include <assert.h>
#include <math.h>

using namespace std;

class SVD{
    
public:
    // data members
    int NUM_USERS = 458293;
    int NUM_MOVIES = 17770;
    
    // member functions
    SVD();
    void Train_model();
    void predict();
    float get_rmse();
    
private:
    // data members
    int NUM_TOTAL = 102416306;
    int NUM_QUAL = 2749898;
    // int NUM_TRAIN = NUM_TOTAL - NUM_QUAL;
    int NUM_LATENT = 20;
    
    int MAX_EPOCHS = 20;
    float eta = 0.004;
    float reg = 0.01;
    vector<vector<int>> base_data;
    vector<vector<int>> valid_data;
    vector<vector<int>> hidden_data;
    vector<vector<int>> probe_data;
    vector<vector<int>> qual_data;
    float tot_avg;
    vector<float> user_bias;
    vector<float> movie_bias;
    vector<vector<float>> U;
    vector<vector<float>> V;
};



// define vector dot product
template <typename T> float dot(const std::vector<T>& a, const std::vector<T>& b)
{
    assert(a.size() == b.size());
    float result = 0;
    for (int i = 0; i < a.size(); i++){\
        result = result + a[i]*b[i];
    }
    return result;
    
}

// define left scalar multiplication of vector
template <typename T> std::vector<T> operator*(const float& c, const std::vector<T>& b)
{
    
    std::vector<T> result = b;
    for( typename vector<T>::iterator it = result.begin(); it != result.end(); ++it){
        *it *= c;
    }
    
    return result;
}

// define right scalar multiplication of vector
template <typename T> std::vector<T> operator*(const std::vector<T>& b, const float& c)
{
    return c*b;
}

// define vector addition
template <typename T> std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b)
{
    
    assert(a.size() == b.size());
    
    std::vector<T> result;
    result.reserve(a.size());
    
    std::transform(a.begin(), a.end(), b.begin(),
                   std::back_inserter(result), std::plus<T>());
    return result;
}

// define vector subtraction
template <typename T> std::vector<T> operator-(const std::vector<T>& a, const std::vector<T>& b)
{
    assert(a.size() == b.size());
    
    std::vector<T> result;
    result.reserve(a.size());
    
    std::transform(a.begin(), a.end(), b.begin(),
                   std::back_inserter(result), std::minus<T>());
    return result;
}



// define constructor
SVD::SVD(){
    
    // Input Data Setup
    ifstream indata, indices;
    int num, index;
    int row, col;
    
    indata.open("/Users/zitaowang/Dropbox/Academics/Caltech/CS156/b/Data/mu/all.dta"); // opens the file
    indices.open("/Users/zitaowang/Dropbox/Academics/Caltech/CS156/b/Data/mu/all.idx");
    
    // Read in Data, initialize base_data
    indata >> num;
    indices >> index;
    row = 0;
    col = 0;
    if(!indata){ // file couldn't be opened
        cerr << "Error: file could not be opened" << endl;
        exit(1);
    }
    
    vector<int> data_vect(4);
    while(!indata.eof()){
        
        for(col = 0; col < 4; col++){
            switch(col){
                case 0:
                    data_vect[0] = num;
                    break;
                case 1:
                    data_vect[1] = num;
                    break;
                case 2:
                    data_vect[3] = num;
                    break;
                case 3:
                    data_vect[2] = num;
                    tot_avg = tot_avg + num;
                    break;
                default:
                    break;
            }
            indata >> num;
        }

        switch(index){
            case 1:
                base_data.push_back(data_vect);
                break;
            case 2:
                valid_data.push_back(data_vect);
                break;
            case 3:
                hidden_data.push_back(data_vect);
                break;
            case 4:
                probe_data.push_back(data_vect);
                break;
            case 5:
                qual_data.push_back(data_vect);
                break;
            default:
                break;
        }
        indices >> index;
        
        if(row % 1000000 == 0){
            cout << "Reading Training Set: " << 100 * (float)row / (float)NUM_TOTAL << " %" << endl;
        }
        
        row += 1;
        
        if(row == NUM_TOTAL){
            break;
        }
    }
    indata.close();
    indices.close();
    tot_avg = (float)tot_avg / base_data.size();
    cout << base_data.size()<<endl;
    cout << valid_data.size()<<endl;
    cout << hidden_data.size()<<endl;
    cout << probe_data.size()<<endl;
    cout << qual_data.size()<<endl;
    
    srand (static_cast <unsigned> (time(0)));
    vector<float> random_vec(NUM_LATENT);
    // initialize user_bias and U
    for(int j = 0; j < NUM_USERS; j++){
        user_bias.push_back(((float)rand() / RAND_MAX)-0.5);
        for(int k = 0; k < NUM_LATENT; k++){
            random_vec[k] = ((float)rand() / RAND_MAX)-0.5;
        }
        U.push_back(random_vec);
    }
    // initialize V
    for(int j = 0; j < NUM_MOVIES; j++){
        movie_bias.push_back(((float)rand() / RAND_MAX)-0.5);
        for(int k = 0; k < NUM_LATENT; k++){
            random_vec[k] = ((float)rand() / RAND_MAX)-0.5;
        }
        V.push_back(random_vec);
    }
}


// define member functions
void SVD::Train_model(){
    for(int epoch = 1; epoch <= MAX_EPOCHS; epoch = epoch + 1){
        vector<float> delta_user(NUM_LATENT), delta_movie(NUM_LATENT);
        float delta_user_bias, delta_movie_bias;
        vector<int> index_vec(base_data.size());
        for(int i = 0; i < base_data.size(); i++){
            index_vec[i] = i;
        }
        random_shuffle(index_vec.begin(), index_vec.end());
        for(int j = 0; j < base_data.size(); j++){
            int data_index = index_vec[j];
            int user = base_data[data_index][0];
            int movie = base_data[data_index][1];
            int rate = base_data[data_index][2];
            
            float prediction = tot_avg + user_bias[user-1] + movie_bias[movie-1] + dot(U[user-1],V[movie-1]);
            delta_user = eta*(reg*U[user-1]-2*(rate - prediction)*V[movie-1]);
            delta_movie = eta*(reg*V[movie-1]-2*(rate - prediction)*U[user-1]);
            delta_user_bias = eta*(-2*(rate - prediction) + 2*reg*user_bias[user-1]);
            delta_movie_bias = eta*(-2*(rate - prediction) + 2*reg*movie_bias[movie-1]);
            
            U[user-1] = U[user-1] - delta_user;
            V[movie-1] = V[movie-1] - delta_movie;
            user_bias[user-1] = user_bias[user-1] - delta_user_bias;
            movie_bias[movie-1] = movie_bias[movie-1] - delta_movie_bias;
            if(j % 1000000 == 0){
                cout << "Epoch percentage: " << 100 * (float)j / (float)base_data.size() << " %" << endl;
                cout << "RMSE = " << get_rmse() << endl;
                cout << "Above Water " << (0.9525 - get_rmse()) / 0.9525 << endl;
            }
        }
        cout << "Epoch " << epoch << endl;
        
    }
    cout << "Done!" << endl;
}


float SVD::get_rmse(){
    float sum = 0;
    for (int i = 0; i < valid_data.size(); ++i){
        int user = valid_data[i][0];
        int movie = valid_data[i][1];
        int rating = valid_data[i][2];
        float prediction = tot_avg + user_bias[user-1] + movie_bias[movie-1] + dot(U[user-1],V[movie-1]);
        sum = sum + (prediction - rating)*(prediction - rating);
    }
    return sqrt(sum/valid_data.size());
}

void SVD::predict(){
    ifstream indata;
    int num;
    int row, col;
    
    indata.open("/Users/zitaowang/Dropbox/Academics/Caltech/CS156/b/Data/mu/qual.dta"); // opens the qual file
    indata >> num;
    row = 0;
    col = 0;
    
    ofstream prediction_file;
    prediction_file.open ("prediction.txt");
    
    while(!indata.eof())
    {
        if (row % (NUM_QUAL/100) == 0) {
            cout << "Predicting Qual: " << (int)(100 * float(row) / float(NUM_QUAL)) << " %" << endl;
        }
        
        int user_id = 0;
        int movie_id = 0;
        for (col = 0; col < 3; col++) {
            switch (col) {
                case 0:
                    user_id = num;
                    break;
                case 1:
                    movie_id = num;
                    break;
                default:
                    break;
            }
            indata >> num;
        }
        row += 1;
        float pred = tot_avg + user_bias[user_id-1] + movie_bias[movie_id-1] + dot(U[user_id-1],V[movie_id-1]);
        prediction_file << std::fixed << std::setprecision(3) << pred << "\n";
    }
    prediction_file.close();
    indata.close();
}



int main(int argc, const char * argv[]) {
    
    SVD svd;
    svd.Train_model();
    svd.predict();
    return 0;
}
