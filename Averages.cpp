//
//  main.cpp
//  Averages
//
//  Created by Rupesh Jeyaram on 4/14/18.
//  Copyright © 2018 Rupesh Jeyaram. All rights reserved.
//

#include <iostream>
#include "Eigen/Eigen"
#include "Eigen/Sparse"
#include <vector>
#include <fstream>
#include <cstdlib>

using namespace std;
using namespace Eigen;

int row, col;

int NUM_TOTAL = 102416306;

int NUM_QUAL = 2749898;
int NUM_TRAIN = NUM_TOTAL - NUM_QUAL;

int NUM_USERS = 458293;
int NUM_MOVIES = 17770;

int LIMIT = NUM_TOTAL;

SparseMatrix<double, RowMajor> m_row(NUM_USERS+1,NUM_MOVIES+1);
SparseMatrix<double, ColMajor> m_col(NUM_USERS+1,NUM_MOVIES+1);

float average_of_user(int id) {
    return (float)m_row.row(id).sum()/(float)m_row.row(id).nonZeros();
}

float average_of_movie(int id) {
    return (float)m_col.col(id).sum()/(float)m_col.col(id).nonZeros();
}

int main(int argc, const char * argv[]) {
    
    // Triplet Setup
    typedef Eigen::Triplet<double> T;
    vector<T> tripletList;
    tripletList.reserve(LIMIT * 4);
    
    // Input Data Setup
    ifstream indata;
    int num;
    
    indata.open("/Users/rupesh/Desktop/mu/all.dta"); // opens the file
    
    // Read in Data, Populate Matrix
    indata >> num;
    row = 0;
    col = 0;
    if(!indata) { // file couldn't be opened
        cerr << "Error: file could not be opened" << endl;
        exit(1);
    }
    
    while(!indata.eof())
    {
        int user_id = 0;
        int movie_id = 0;
        int date = 0;
        int rating = 0;
        
        for (col = 0; col < 4; col++) {
            switch (col) {
                case 0:
                    user_id = num;
                    break;
                case 1:
                    movie_id = num;
                    break;
                case 2:
                    date = num;
                    break;
                case 3:
                    rating = num;
                    break;
                default:
                    break;
            }
            indata >> num;
        }
        
        tripletList.push_back(T(user_id,movie_id,rating));
        
        if (row % 1000000 == 0 ) {
            cout << "Reading Training Set: " << 100 * (float)row / (float)NUM_TOTAL << " %" << endl;
        }
        
        row += 1;
        
        if (row == LIMIT) {
            break;
        }
    }
    indata.close();

    cout << tripletList.size() << endl;
    cout << "Converting to Row Major Sparse Matrix..." << endl;
    m_row.setFromTriplets(tripletList.begin(), tripletList.end());
    cout << "Converting to Column Major Sparse Matrix..." << endl;
    m_col.setFromTriplets(tripletList.begin(), tripletList.end());
    cout << "Done" << endl;
    

    indata.open("/Users/rupesh/Desktop/mu/qual.dta"); // opens the qual file
    indata >> num;
    row = 0;
    col = 0;

    remove("predictions.txt");
    ofstream prediction_file;
    prediction_file.open ("predictions.txt");
    
    vector<float> average_movies;
    
    for (int i=0; i < NUM_MOVIES; i ++) {
        average_movies.push_back(average_of_movie(i));
    }
    
    vector<float> average_users;
    
    for (int i=0; i < NUM_USERS; i ++) {
        average_users.push_back(average_of_user(i));
    }

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
        prediction_file << 0.5 * (average_movies[movie_id] + average_users[user_id]) << "\n";
    }

    prediction_file.close();
    indata.close();
}
