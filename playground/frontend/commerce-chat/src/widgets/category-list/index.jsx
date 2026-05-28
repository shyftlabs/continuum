import React, { useState, useMemo, useEffect } from "react";
import { createRoot } from "react-dom/client";
import { ChevronDown, ChevronRight } from "lucide-react";
import { Badge } from "@openai/apps-sdk-ui/components/Badge";
import { useWidgetProps } from "../use-widget-props";

const fallbackCategories = [
  {
    id: "1",
    name: "Electronics",
    parent_id: null,
    image: "https://www.gadgetsalvation.com/blog/wp-content/uploads/2023/12/Untitled-1-1024x683.jpeg",
  },
  {
    id: "2",
    name: "Mobiles",
    parent_id: "1",
    image: "https://www.gadgetsalvation.com/blog/wp-content/uploads/2023/12/Untitled-1-1024x683.jpeg",
  },
  {
    id: "3",
    name: "Laptops",
    parent_id: "1",
    image: "https://www.gadgetsalvation.com/blog/wp-content/uploads/2023/12/Untitled-1-1024x683.jpeg",
  },
  {
    id: "4",
    name: "Audio",
    parent_id: "1",
    image: "https://www.gadgetsalvation.com/blog/wp-content/uploads/2023/12/Untitled-1-1024x683.jpeg",
  },
  {
    id: "5",
    name: "Home Appliances",
    parent_id: null,
    image: "https://www.gadgetsalvation.com/blog/wp-content/uploads/2023/12/Untitled-1-1024x683.jpeg",
  },
];

function CategoryItem({ category, children, level = 0, isExpanded, onToggle }) {
  const hasChildren = children && children.length > 0;
  const indent = level * 24;

  return (
    <div>
      <div
        className="flex items-center gap-3 p-3 border-b border-black/10 last:border-0 bg-white hover:bg-gray-50 transition-colors cursor-pointer"
        style={{ paddingLeft: `${12 + indent}px` }}
        onClick={() => hasChildren && onToggle && onToggle(category.id)}
      >
        {/* Expand/Collapse Icon */}
        {hasChildren && (
          <div className="flex-shrink-0 w-5 h-5 flex items-center justify-center">
            {isExpanded ? (
              <ChevronDown className="h-4 w-4 text-black/60" />
            ) : (
              <ChevronRight className="h-4 w-4 text-black/60" />
            )}
          </div>
        )}
        {!hasChildren && <div className="flex-shrink-0 w-5 h-5" />}

        {/* Category Image */}
        <div className="w-16 h-16 flex-shrink-0 overflow-hidden rounded-lg bg-gray-100">
          <img
            src={category.image || category.thumbnail || category.cover || category.cover_image || ""}
            alt={category.name || "Category"}
            className="w-full h-full object-cover"
            loading="lazy"
          />
        </div>

        {/* Category Info */}
        <div className="flex-1 min-w-0">
          <h3 className="text-base font-semibold text-black truncate mb-1">
            {category.name || category.title || "Category"}
          </h3>
          <div className="flex items-center gap-2">
            <Badge color="info" size="sm">
              {hasChildren ? `${children.length} subcategories` : "Category"}
            </Badge>
           
          </div>
        </div>
      </div>

      {/* Children */}
      {hasChildren && isExpanded && (
        <div>
          {children.map((child) => (
            <CategoryItem
              key={child.id}
              category={child}
              children={child.children}
              level={level + 1}
              isExpanded={child.isExpanded}
              onToggle={onToggle}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function App() {
  const widgetProps = useWidgetProps(() => ({}));
  const [itemsData, setItemsData] = useState(null);
  const [expandedCategories, setExpandedCategories] = useState(new Set());

  // Poll for data injection
  useEffect(() => {
    const MAX_ATTEMPTS = 60;
    const DELAY = 150;
    let attempts = 0;

    function pullData() {
      const openai = window.openai || {};
      const toolOutput = openai.toolOutput;

      console.log("CategoryList - pullData attempt:", attempts);
      console.log("CategoryList - toolOutput:", toolOutput);

      if (toolOutput && typeof toolOutput === 'object' && Object.keys(toolOutput).length > 0) {
        console.log("CategoryList - Found data in toolOutput:", toolOutput);
        setItemsData(toolOutput);
        return true;
      }

      if (widgetProps && typeof widgetProps === 'object' && Object.keys(widgetProps).length > 0) {
        console.log("CategoryList - Found data in widgetProps:", widgetProps);
        setItemsData(widgetProps);
        return true;
      }

      // Also check window.widgetProps if available
      if (window.widgetProps && typeof window.widgetProps === 'object' && Object.keys(window.widgetProps).length > 0) {
        console.log("CategoryList - Found data in window.widgetProps:", window.widgetProps);
        const data = window.widgetProps.structuredContent || window.widgetProps;
        setItemsData(data);
        return true;
      }

      return false;
    }

    function hydrate() {
      if (pullData()) {
        return;
      }
      if (attempts++ < MAX_ATTEMPTS) {
        setTimeout(hydrate, DELAY);
      } else {
        console.warn("CategoryList - No data found after", MAX_ATTEMPTS, "attempts, using fallback");
        setItemsData({});
      }
    }

    hydrate();
  }, [widgetProps]);

  // Build hierarchical category structure
  const categoryTree = useMemo(() => {
    if (!itemsData) {
      return buildCategoryTree(fallbackCategories, expandedCategories);
    }

    console.log("CategoryList - Processing itemsData:", itemsData);

    let rawCategories = null;

    // Check for different data formats
    if (Array.isArray(itemsData)) {
      rawCategories = itemsData;
      console.log("CategoryList - Found direct array:", rawCategories.length);
    } else if (Array.isArray(itemsData.items)) {
      rawCategories = itemsData.items;
      console.log("CategoryList - Found items array:", rawCategories.length);
    } else if (Array.isArray(itemsData.categories)) {
      rawCategories = itemsData.categories;
      console.log("CategoryList - Found categories array:", rawCategories.length);
    } else if (Array.isArray(itemsData.albums)) {
      rawCategories = itemsData.albums;
      console.log("CategoryList - Found albums array:", rawCategories.length);
    }

    if (rawCategories && Array.isArray(rawCategories) && rawCategories.length > 0) {
      // Normalize categories - ensure consistent ID types
      const normalized = rawCategories.map((item, index) => {
        const id = String(item.id || item.category_id || `category-${index}`);
        const parentId = item.parent_id !== null && item.parent_id !== undefined 
          ? String(item.parent_id) 
          : null;
        
        return {
          id: id,
          name: item.name || item.title || item.category_name || "Category",
          parent_id: parentId,
          image: item.image || item.image_url || item.thumbnail || item.cover || item.cover_image || "",
        };
      });

      return buildCategoryTree(normalized, expandedCategories);
    }

    // Fallback to default categories
    console.log("CategoryList - Using fallback categories");
    return buildCategoryTree(fallbackCategories, expandedCategories);
  }, [itemsData, expandedCategories]);

  // Build tree structure from flat array
  function buildCategoryTree(categories, expandedSet) {
    const categoryMap = new Map();
    const rootCategories = [];

    // First pass: create all category objects with normalized IDs
    categories.forEach((cat) => {
      const normalizedId = String(cat.id);
      categoryMap.set(normalizedId, {
        ...cat,
        id: normalizedId,
        children: [],
        isExpanded: expandedSet.has(normalizedId),
      });
    });

    // Second pass: build parent-child relationships
    categories.forEach((cat) => {
      const normalizedId = String(cat.id);
      const category = categoryMap.get(normalizedId);
      
      // Normalize parent_id for comparison
      const normalizedParentId = cat.parent_id !== null && cat.parent_id !== undefined 
        ? String(cat.parent_id) 
        : null;
      
      if (normalizedParentId && categoryMap.has(normalizedParentId)) {
        const parent = categoryMap.get(normalizedParentId);
        parent.children.push(category);
      } else {
        rootCategories.push(category);
      }
    });

    return rootCategories;
  }

  const handleToggleCategory = (categoryId) => {
    setExpandedCategories((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(categoryId)) {
        newSet.delete(categoryId);
      } else {
        newSet.add(categoryId);
      }
      return newSet;
    });
  };

  const totalCategories = useMemo(() => {
    const count = (categories) => {
      return categories.reduce((sum, cat) => {
        return sum + 1 + (cat.children ? count(cat.children) : 0);
      }, 0);
    };
    return count(categoryTree);
  }, [categoryTree]);

  if (categoryTree.length === 0) {
    return (
      <section className="mx-auto max-w-6xl space-y-5 px-4 py-8">
        <div className="text-center py-12">
          <p className="text-black/60 text-lg">No categories available.</p>
        </div>
      </section>
    );
  }

  return (
    <section className="mx-auto max-w-6xl space-y-5 px-4 py-8">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-black/60 text-sm">Categories</p>
          <h1 className="text-2xl font-semibold text-black">Category List</h1>
        </div>
        <div className="text-sm text-black/60">
          {totalCategories} {totalCategories === 1 ? 'category' : 'categories'}
        </div>
      </div>

      <div className="bg-white rounded-2xl border border-black/10 shadow-lg overflow-hidden">
        <div className="overflow-x-auto">
          <div className="min-w-full">
            {categoryTree.map((category) => (
              <CategoryItem
                key={category.id}
                category={category}
                children={category.children}
                level={0}
                isExpanded={expandedCategories.has(category.id)}
                onToggle={handleToggleCategory}
              />
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

export { App };

const rootElement = document.getElementById("category-list-root");
if (rootElement) {
  createRoot(rootElement).render(<App />);
} else {
  console.error("CategoryList - Root element 'category-list-root' not found!");
}
