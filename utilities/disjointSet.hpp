#pragma once

#include <map>
#include <utility>

namespace memopt {

/**
 * Efficient implementation of the Disjoint-Set (Union-Find) data structure
 * 
 * This class implements the Disjoint-Set data structure, which efficiently
 * maintains a collection of non-overlapping sets and supports two key operations:
 * 1. Union - merge two sets together
 * 2. Find - determine which set an element belongs to
 * 
 * The implementation uses:
 * - Path compression during find operations to flatten trees
 * - Union by size optimization to keep trees balanced
 * 
 * These optimizations ensure near-constant time operations, making this
 * data structure exceptionally efficient for grouping elements.
 * 
 * Primary use cases in this project include:
 * - Grouping nodes that execute concurrently
 * - Merging nodes with the same annotations
 * - Identifying connected components in the CUDA graph
 * 
 * @tparam KeyType The type of elements stored in the sets
 */
template <typename KeyType>
class DisjointSet {
 public:
  DisjointSet() = default;
  
  /**
   * Finds the root (representative element) of the set containing the given element
   * 
   * This method implements the "find" operation of the disjoint-set data structure
   * with path compression optimization. It:
   * 1. Creates a new singleton set if the element doesn't exist yet
   * 2. Recursively finds the root of the set
   * 3. Performs path compression by updating parent pointers directly to the root
   * 
   * Path compression ensures that future find operations for the same element
   * or its siblings will be faster, approaching constant time.
   * 
   * @param u The element to find the set root for
   * @return The root element of the set containing u
   */
  KeyType findRoot(KeyType u) {
    // Try to find the element in the parent map
    auto it = this->parent.find(u);
    
    // If not found, create a new singleton set for this element
    if (it == this->parent.end()) {
      const auto [insertedIt, success] = this->parent.insert({u, u});
      it = insertedIt;
      // Initialize the set size to 1
      this->size[u] = 1;
    }

    // Get the parent of this element
    auto p = it->second;
    
    // If this element is its own parent, it's a root
    if (p == u) {
      return u;
    } else {
      // Not a root - perform path compression:
      // Recursively find the true root
      p = findRoot(p);
      // Update this element's parent to point directly to the root (compress)
      it->second = p;
      // Return the root
      return this->findRoot(p);
    }
  }

  /**
   * Merges two sets in the disjoint-set data structure
   * 
   * This method combines two separate sets into a single set by making one root
   * point to the other. It implements the "union by size" optimization to keep
   * the tree height small, ensuring efficient future operations.
   * 
   * The method:
   * 1. Finds the root elements of both sets
   * 2. If they're already in the same set, does nothing
   * 3. Makes the smaller tree a subtree of the larger tree to minimize tree height
   * 4. Updates the size of the merged set
   * 
   * After this operation, elements from both original sets will have the same root,
   * indicating they now belong to the same logical group.
   * 
   * @param u First element to union
   * @param v Second element to union
   */
  void unionUnderlyingSets(KeyType u, KeyType v) {
    // Step 1: Find the roots of both elements
    // This ensures we're working with the set representatives
    u = this->findRoot(u);
    v = this->findRoot(v);
    
    // If both elements are already in the same set, nothing to do
    if (u == v) return;

    // Step 2: Apply "union by size" optimization
    // Always make the smaller tree a subtree of the larger tree
    // This keeps the resulting tree height minimal for efficient find operations
    if (this->size[u] < this->size[v]) std::swap(u, v);
    
    // Step 3: Make 'v' root point to 'u' root
    // This effectively combines the two trees
    this->parent[v] = u;
    
    // Step 4: Update the size of the merged tree
    // The new size is the sum of both original tree sizes
    this->size[u] = this->size[u] + this->size[v];
  }

  /**
   * Gets the number of elements in the set containing the given element
   * 
   * This method returns the size of the set that contains the specified element.
   * It first finds the root of the set and then returns the size stored for that root.
   * 
   * @param u The element whose set size to retrieve
   * @return The number of elements in the set containing u
   */
  size_t getSetSize(KeyType u) {
    return this->size[this->findRoot(u)];
  }

  /**
   * Gets the total number of distinct sets in the disjoint-set data structure
   * 
   * This method counts the total number of roots (parent nodes that point to themselves),
   * which represents the number of distinct sets currently in the data structure.
   * 
   * @return The number of distinct sets
   */
  size_t getNumSets() {
    size_t count = 0;
    for (const auto& [key, value] : parent) {
      if (key == value) {
        count++;
      }
    }
    return count;
  }

 private:
  // Maps each element to its parent element in the tree structure
  std::map<KeyType, KeyType> parent;
  
  // Maps each root element to the size of its set
  std::map<KeyType, size_t> size;
};

}  // namespace memopt
