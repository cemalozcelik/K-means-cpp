#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <opencv2/opencv.hpp>

// Node class for data points

class Node {
public:
    Node(double xCoordinate, double yCoordinate, int identity);
    double getX() const;
    double getY() const;
    int getIdentity() const;
    void print() const;

private:
    double x;
    double y;
    int identity;
};

Node::Node(double xCoordinate, double yCoordinate, int identity)
    : x(xCoordinate), y(yCoordinate), identity(identity) {}

double Node::getX() const {
    return x;
}

double Node::getY() const {
    return y;
}

int Node::getIdentity() const {
    return identity;
}

void Node::print() const {
    std::cout << "Node " << getIdentity() << " (" << getX() << ", " << getY() << ")\n";
}

// Clustering class for managing data points and centroids
class Clustering {
public:
    Clustering(int numClusters);
    void loadNodesFromFile(const std::string& fileName);
    const std::vector<Node>& getNodes() const;
    int getK() const;

private:
    int numClusters;
    std::vector<Node> nodes;
    int k;
};

Clustering::Clustering(int numClusters) : numClusters(numClusters) {}

void Clustering::loadNodesFromFile(const std::string& fileName) {
    std::ifstream file(fileName);
    if (!file) {
        std::cerr << "Error opening file: " << fileName << "\n";
        return;
    }

    double x, y;
    int identity = 1;
    while (file >> x >> y) {
        nodes.emplace_back(x, y, identity++);
    }

    file.close();
}

const std::vector<Node>& Clustering::getNodes() const {
    return nodes;
}

int Clustering::getK() const {
    return numClusters;
}

// KMeansPP class for applying KMeans++ algorithm
class KMeansPP {
public:
    KMeansPP(const Clustering& clustering);
    void apply(int numIterations);    

private:
    std::default_random_engine generator; 
    const Clustering& clustering;
    std::vector<Node> centroids;
    std::vector<std::vector<Node>> clusters;

    void initializeFirstCentroid();
    void initializeRestCentroids();
    double findTotalDistance();
    void findDistances();
    void performIteration();
    Node calculateClusterCentroid(const std::vector<Node>& cluster);
    double calculateEuclidean(double x1, double y1, double x2, double y2);
    void printCentroids() const;
    void printClusters() const;
    void plotScatter() const;
    
};

KMeansPP::KMeansPP(const Clustering& clustering)
    : clustering(clustering), generator(std::random_device()()) {}
void KMeansPP::apply(int numIterations) {
    int iterations = 0;
    initializeFirstCentroid();
    initializeRestCentroids();
    std::cout << "Enter the number of iterations: ";
    std::cin >> iterations;

    for (int i = 0; i < iterations; i++) {
        std::cout << "Iteration " << i + 1 << "\n\n";
        findDistances();
        performIteration();
        plotScatter();
    }
    
}

void KMeansPP::initializeFirstCentroid() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, clustering.getNodes().size() - 1);
    int randomIndex = dist(gen);
    centroids.push_back(clustering.getNodes()[randomIndex]);
}

void KMeansPP::initializeRestCentroids() {
    while (centroids.size() < clustering.getK()) {
        double totalDistance = findTotalDistance();
        std::vector<double> probabilities;

        for (const auto& node : clustering.getNodes()) {
            double minDistanceToCentroid = std::numeric_limits<double>::max();
            for (const auto& centroid : centroids) {
                double distance = calculateEuclidean(centroid.getX(), centroid.getY(), node.getX(), node.getY());
                minDistanceToCentroid = std::min(minDistanceToCentroid, distance);
            }

            double probability = (minDistanceToCentroid * minDistanceToCentroid) / totalDistance;
            probabilities.push_back(probability);
        }

        std::discrete_distribution<> distribution(probabilities.begin(), probabilities.end());
        int randomIndex = distribution(generator);
        centroids.push_back(clustering.getNodes()[randomIndex]);
    }
}

void KMeansPP::performIteration() {
    clusters.clear();
    clusters.resize(clustering.getK());

    for (const auto& node : clustering.getNodes()) {
        double minDistance = std::numeric_limits<double>::max();
        int closestCentroidIndex = -1;

        for (size_t i = 0; i < centroids.size(); ++i) {
            double distance = calculateEuclidean(centroids[i].getX(), centroids[i].getY(), node.getX(), node.getY());
            if (distance < minDistance) {
                minDistance = distance;
                closestCentroidIndex = static_cast<int>(i);
            }
        }

        if (closestCentroidIndex != -1) {
            clusters[closestCentroidIndex].push_back(node);
        }
    }

    for (size_t i = 0; i < centroids.size(); ++i) {
        if (!clusters[i].empty()) {
            centroids[i] = calculateClusterCentroid(clusters[i]);
        }
    }

    printCentroids();
}



double KMeansPP::findTotalDistance() {
    double totalDistance = 0.0;

    for (const auto& node : clustering.getNodes()) {
        for (const auto& centroid : centroids) {
            totalDistance += std::pow(calculateEuclidean(centroid.getX(), centroid.getY(), node.getX(), node.getY()), 2);
        }
    }
    return totalDistance;
}

void KMeansPP::findDistances() {
    clusters.clear();
    clusters.resize(clustering.getK());

    for (const auto& node : clustering.getNodes()) {
        std::vector<double> distances;

        for (const auto& centroid : centroids) {
            distances.push_back(calculateEuclidean(centroid.getX(), centroid.getY(), node.getX(), node.getY()));
        }

        int minElementIndex = std::min_element(distances.begin(), distances.end()) - distances.begin();
        clusters[minElementIndex].push_back(node);
    }
}


Node KMeansPP::calculateClusterCentroid(const std::vector<Node>& cluster) {
    double totalX = 0.0;
    double totalY = 0.0;

    for (const auto& node : cluster) {
        totalX += node.getX();
        totalY += node.getY();
    }

    double averageX = totalX / cluster.size();
    double averageY = totalY / cluster.size();

    return Node(averageX, averageY, -1); // Use a placeholder identity
}

double KMeansPP::calculateEuclidean(double x1, double y1, double x2, double y2) {
    return std::sqrt(std::pow(x1 - x2, 2) + std::pow(y1 - y2, 2));
}

void KMeansPP::printCentroids() const {
    for (const auto& centroid : centroids) {
        centroid.print();
    }
}

void KMeansPP::printClusters() const {
    for (size_t i = 0; i < clusters.size(); ++i) {
        std::cout << "Cluster " << i + 1 << ":\n";
        for (const auto& node : clusters[i]) {
            std::cout << "Node " << node.getIdentity() << " (" << node.getX() << ", " << node.getY() << ")\n";
        }
        std::cout << "\n";
    }
}
void KMeansPP::plotScatter() const {
    // Create a blank canvas to draw the scatter plot
    cv::Mat scatterPlot(800, 800, CV_8UC3, cv::Scalar(255, 255, 255));

    // Scale data points to fit within the canvas size
    double max_x = 0, max_y = 0;
    for (const auto& node : clustering.getNodes()) {
        max_x = std::max(max_x, node.getX());
        max_y = std::max(max_y, node.getY());
    }

    for (size_t i = 0; i < centroids.size(); ++i) {
        max_x = std::max(max_x, centroids[i].getX());
        max_y = std::max(max_y, centroids[i].getY());
    }

    double scale_factor_x = scatterPlot.cols / (max_x + 10);
    double scale_factor_y = scatterPlot.rows / (max_y + 10);


    // Define a vector of distinct colors for clusters
    std::vector<cv::Scalar> clusterColors = {
        cv::Scalar(255, 0, 0),   // Red
        cv::Scalar(0, 255, 0),   // Green
        cv::Scalar(0, 0, 255),   // Blue
        cv::Scalar(255, 255, 0), // Yellow
        // Add more colors as needed
    };

    // Draw data points with cluster-specific colors
    for (size_t i = 0; i < clusters.size(); ++i) {
        for (const auto& node : clusters[i]) {
            cv::Point point(node.getX() * scale_factor_x, node.getY() * scale_factor_y);
            cv::circle(scatterPlot, point, 3, clusterColors[i % clusterColors.size()], -1);
        }
    }

    // Draw centroids and label them
    for (size_t i = 0; i < centroids.size(); ++i) {
        cv::Point point(centroids[i].getX() * scale_factor_x, centroids[i].getY() * scale_factor_y);
        cv::circle(scatterPlot, point, 7, cv::Scalar(0, 0, 0), -1); // Black circle for centroids
        
        // Print labels indicating the centers
        std::stringstream label;
        label << "Center " << i + 1;
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(label.str(), cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseline);
        cv::putText(scatterPlot, label.str(), cv::Point(point.x - textSize.width / 2, point.y - 10), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1); // Black label for centroids
    }

    // Save the scatter plot image
    cv::imwrite("scatter_plot.png", scatterPlot);

    // Display the scatter plot
    cv::imshow("Scatter Plot", scatterPlot);
    cv::waitKey(0);
    cv::destroyAllWindows();
}



int main() {
    int numClusters, numIterations;
    std::cout << "Enter the number of clusters: ";
    std::cin >> numClusters;
    std::cout << "Enter the number of iterations: ";
    std::cin >> numIterations;

    Clustering clustering(numClusters);
    clustering.loadNodesFromFile("50_points.txt");

    KMeansPP kmeans(clustering);
    kmeans.apply(numIterations);

    return 0;
}

