% HW1 | Principle Component Analysis (PCA) and K-Nearest Neighbour
% Clustering
%
% AUTHOR: Zeyuan Hu <iamzeyuanhu@utexas.edu | EID: zh4378>
%
% INSTRUCTIONS
% ------------
%
%

%% Initialization
clear; close all; clc

%% Global variables we want to use
NUM_TRAIN = 500;  % number of training examples

%% ============== Part 1: Load Example Dataset =======================
%
%
%

% The following command loads the dataset. You should now have
% the variable trainLabels, testLabels, testImages, trainImages in your
% environment.
load('digits.mat');

% We want to convert each image in our trainImages variable into a column
% vector of a matrix.
[m,n,k,j] = size(trainImages); 
