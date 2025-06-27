"""
Network Topology Cache for Supply Chain Network.

This module provides caching mechanisms to eliminate redundant network traversals
during agent operations, significantly improving performance.
"""

import logging
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
import time

from disruptsc.model.profiling_utils import profile_function, profile_method


class NetworkTopologyCache:
    """
    Caches supply chain network topology to eliminate redundant traversals.
    
    This cache pre-computes and stores network relationships to avoid repeated
    graph traversals during agent operations like product reception, purchase
    planning, and order processing.
    """
    
    def __init__(self, sc_network=None):
        """
        Initialize the topology cache.
        
        Parameters
        ----------
        sc_network : ScNetwork, optional
            Supply chain network to cache. Can be set later with build_cache()
        """
        # Core topology caches
        self.supplier_map: Dict[str, List[Tuple]] = {}      # agent_id -> [(supplier, edge_data), ...]
        self.customer_map: Dict[str, List[Tuple]] = {}      # agent_id -> [(customer, edge_data), ...]
        self.link_cache: Dict[Tuple[str, str], Any] = {}   # (supplier_id, buyer_id) -> commercial_link
        
        # Network metadata
        self.agent_types: Dict[str, str] = {}               # agent_id -> agent_type
        self.agent_sectors: Dict[str, str] = {}             # agent_id -> sector
        self.network_structure: Dict[str, Any] = {}         # Network summary statistics
        
        # Cache management
        self.cache_built = False
        self.cache_build_time = 0.0
        self.cache_version = 0
        self.hit_count = 0
        self.miss_count = 0
        
        # Build cache if network provided
        if sc_network is not None:
            self.build_cache(sc_network)
    
    @profile_method
    def build_cache(self, sc_network) -> None:
        """
        Build the topology cache from the supply chain network.
        
        Parameters
        ----------
        sc_network : ScNetwork
            Supply chain network to cache
        """
        start_time = time.perf_counter()
        logging.info("Building network topology cache...")
        
        # Clear existing cache
        self.clear_cache()
        
        # Cache supplier relationships
        self._build_supplier_map(sc_network)
        
        # Cache customer relationships  
        self._build_customer_map(sc_network)
        
        # Cache commercial links
        self._build_link_cache(sc_network)
        
        # Cache agent metadata
        self._build_agent_metadata(sc_network)
        
        # Cache network structure info
        self._build_network_structure(sc_network)
        
        # Finalize cache
        self.cache_built = True
        self.cache_build_time = time.perf_counter() - start_time
        self.cache_version += 1
        
        logging.info(f"Network topology cache built in {self.cache_build_time:.3f}s")
        self._log_cache_statistics()
    
    def _build_supplier_map(self, sc_network) -> None:
        """Build mapping from agents to their suppliers."""
        for agent in sc_network.nodes:
            agent_id = agent.pid
            suppliers = []
            
            for supplier, _ in sc_network.in_edges(agent):
                edge_data = sc_network[supplier][agent]
                suppliers.append((supplier, edge_data))
            
            self.supplier_map[agent_id] = suppliers
    
    def _build_customer_map(self, sc_network) -> None:
        """Build mapping from agents to their customers."""
        for agent in sc_network.nodes:
            agent_id = agent.pid
            customers = []
            
            for _, customer in sc_network.out_edges(agent):
                edge_data = sc_network[agent][customer]
                customers.append((customer, edge_data))
            
            self.customer_map[agent_id] = customers
    
    def _build_link_cache(self, sc_network) -> None:
        """Build cache of all commercial links."""
        for supplier, buyer in sc_network.edges:
            link_key = (supplier.pid, buyer.pid)
            commercial_link = sc_network[supplier][buyer]['object']
            self.link_cache[link_key] = commercial_link
    
    def _build_agent_metadata(self, sc_network) -> None:
        """Build cache of agent metadata."""
        for agent in sc_network.nodes:
            self.agent_types[agent.pid] = agent.agent_type
            if hasattr(agent, 'sector'):
                self.agent_sectors[agent.pid] = agent.sector
            elif hasattr(agent, 'region_sector'):
                self.agent_sectors[agent.pid] = agent.region_sector
    
    def _build_network_structure(self, sc_network) -> None:
        """Build network structure summary."""
        agent_type_counts = defaultdict(int)
        sector_counts = defaultdict(int)
        
        for agent in sc_network.nodes:
            agent_type_counts[agent.agent_type] += 1
            if hasattr(agent, 'sector'):
                sector_counts[agent.sector] += 1
        
        self.network_structure = {
            'num_agents': len(sc_network.nodes),
            'num_links': len(sc_network.edges),
            'agent_type_counts': dict(agent_type_counts),
            'sector_counts': dict(sector_counts),
            'avg_suppliers_per_agent': len(sc_network.edges) / len(sc_network.nodes) if sc_network.nodes else 0
        }
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.supplier_map.clear()
        self.customer_map.clear()
        self.link_cache.clear()
        self.agent_types.clear()
        self.agent_sectors.clear()
        self.network_structure.clear()
        
        self.cache_built = False
        self.hit_count = 0
        self.miss_count = 0
    
    def invalidate_cache(self) -> None:
        """Mark cache as invalid (needs rebuilding)."""
        self.cache_built = False
        logging.warning("Network topology cache invalidated - needs rebuilding")
    
    # ===== CACHED ACCESS METHODS =====
    
    def get_suppliers(self, agent_id: str) -> List[Tuple]:
        """
        Get cached suppliers for an agent.
        
        Parameters
        ----------
        agent_id : str
            Agent identifier
            
        Returns
        -------
        List[Tuple]
            List of (supplier_agent, edge_data) tuples
        """
        if not self.cache_built:
            raise RuntimeError("Cache not built - call build_cache() first")
        
        if agent_id in self.supplier_map:
            self.hit_count += 1
            return self.supplier_map[agent_id]
        else:
            self.miss_count += 1
            return []
    
    def get_customers(self, agent_id: str) -> List[Tuple]:
        """
        Get cached customers for an agent.
        
        Parameters
        ----------
        agent_id : str
            Agent identifier
            
        Returns
        -------
        List[Tuple]
            List of (customer_agent, edge_data) tuples
        """
        if not self.cache_built:
            raise RuntimeError("Cache not built - call build_cache() first")
        
        if agent_id in self.customer_map:
            self.hit_count += 1
            return self.customer_map[agent_id]
        else:
            self.miss_count += 1
            return []
    
    def get_commercial_link(self, supplier_id: str, buyer_id: str):
        """
        Get cached commercial link between two agents.
        
        Parameters
        ----------
        supplier_id : str
            Supplier agent identifier
        buyer_id : str
            Buyer agent identifier
            
        Returns
        -------
        CommercialLink or None
            Commercial link object if exists, None otherwise
        """
        if not self.cache_built:
            raise RuntimeError("Cache not built - call build_cache() first")
        
        link_key = (supplier_id, buyer_id)
        if link_key in self.link_cache:
            self.hit_count += 1
            return self.link_cache[link_key]
        else:
            self.miss_count += 1
            return None
    
    def get_agent_type(self, agent_id: str) -> Optional[str]:
        """Get cached agent type."""
        if agent_id in self.agent_types:
            self.hit_count += 1
            return self.agent_types[agent_id]
        else:
            self.miss_count += 1
            return None
    
    def get_agent_sector(self, agent_id: str) -> Optional[str]:
        """Get cached agent sector."""
        if agent_id in self.agent_sectors:
            self.hit_count += 1
            return self.agent_sectors[agent_id]
        else:
            self.miss_count += 1
            return None
    
    def has_supplier_relationship(self, supplier_id: str, buyer_id: str) -> bool:
        """Check if supplier relationship exists (cached)."""
        return (supplier_id, buyer_id) in self.link_cache
    
    def get_suppliers_by_sector(self, agent_id: str, sector: str) -> List[Tuple]:
        """Get suppliers of a specific sector for an agent."""
        suppliers = self.get_suppliers(agent_id)
        return [
            (supplier, edge_data) for supplier, edge_data in suppliers
            if self.get_agent_sector(supplier.pid) == sector
        ]
    
    def get_all_links_for_agent(self, agent_id: str) -> Dict[str, Any]:
        """
        Get all commercial links for an agent (both incoming and outgoing).
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with 'incoming' and 'outgoing' link lists
        """
        incoming_links = []
        outgoing_links = []
        
        # Incoming links (agent as buyer)
        for supplier, edge_data in self.get_suppliers(agent_id):
            link = self.get_commercial_link(supplier.pid, agent_id)
            if link:
                incoming_links.append((supplier.pid, link))
        
        # Outgoing links (agent as supplier)
        for customer, edge_data in self.get_customers(agent_id):
            link = self.get_commercial_link(agent_id, customer.pid)
            if link:
                outgoing_links.append((customer.pid, link))
        
        return {
            'incoming': incoming_links,
            'outgoing': outgoing_links
        }
    
    # ===== BULK OPERATIONS =====
    
    @profile_function("Bulk Supplier Lookup")
    def get_suppliers_bulk(self, agent_ids: List[str]) -> Dict[str, List[Tuple]]:
        """Get suppliers for multiple agents efficiently."""
        return {
            agent_id: self.get_suppliers(agent_id)
            for agent_id in agent_ids
        }
    
    @profile_function("Bulk Link Lookup")
    def get_links_bulk(self, agent_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], Any]:
        """Get commercial links for multiple agent pairs efficiently."""
        return {
            (supplier_id, buyer_id): self.get_commercial_link(supplier_id, buyer_id)
            for supplier_id, buyer_id in agent_pairs
        }
    
    def collect_all_deliveries(self, agent_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Collect all delivery information for agents in a single operation.
        
        This replaces multiple network traversals with cached lookups.
        """
        all_deliveries = defaultdict(dict)
        
        for agent_id in agent_ids:
            suppliers = self.get_suppliers(agent_id)
            
            for supplier, edge_data in suppliers:
                link = self.get_commercial_link(supplier.pid, agent_id)
                if link and link.delivery > 0:
                    all_deliveries[agent_id][link.product] = link.delivery
        
        return dict(all_deliveries)
    
    def collect_all_orders(self, agent_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """Collect all order information for agents in a single operation."""
        all_orders = defaultdict(dict)
        
        for agent_id in agent_ids:
            suppliers = self.get_suppliers(agent_id)
            
            for supplier, edge_data in suppliers:
                link = self.get_commercial_link(supplier.pid, agent_id)
                if link:
                    all_orders[agent_id][supplier.pid] = link.order
        
        return dict(all_orders)
    
    # ===== CACHE STATISTICS AND MONITORING =====
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'cache_built': self.cache_built,
            'cache_version': self.cache_version,
            'build_time': self.cache_build_time,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'network_structure': self.network_structure.copy(),
            'cache_sizes': {
                'suppliers': len(self.supplier_map),
                'customers': len(self.customer_map),
                'links': len(self.link_cache),
                'agent_types': len(self.agent_types),
                'agent_sectors': len(self.agent_sectors)
            }
        }
    
    def _log_cache_statistics(self) -> None:
        """Log cache build statistics."""
        stats = self.get_cache_statistics()
        logging.info(f"Cache statistics:")
        logging.info(f"  - Agents: {stats['network_structure']['num_agents']}")
        logging.info(f"  - Links: {stats['network_structure']['num_links']}")
        logging.info(f"  - Avg suppliers per agent: {stats['network_structure']['avg_suppliers_per_agent']:.1f}")
        logging.info(f"  - Agent types: {stats['network_structure']['agent_type_counts']}")
    
    def print_cache_summary(self) -> None:
        """Print comprehensive cache summary."""
        stats = self.get_cache_statistics()
        
        print("\n" + "="*60)
        print("📊 NETWORK TOPOLOGY CACHE SUMMARY")
        print("="*60)
        print(f"Cache Status:        {'✅ Built' if stats['cache_built'] else '❌ Not Built'}")
        print(f"Cache Version:       {stats['cache_version']}")
        print(f"Build Time:          {stats['build_time']:.3f}s")
        print(f"Hit Rate:            {stats['hit_rate']:.1%} ({stats['hit_count']}/{stats['hit_count'] + stats['miss_count']})")
        
        print(f"\n📈 Network Structure:")
        print(f"Agents:              {stats['network_structure']['num_agents']:,}")
        print(f"Commercial Links:    {stats['network_structure']['num_links']:,}")
        print(f"Avg Suppliers/Agent: {stats['network_structure']['avg_suppliers_per_agent']:.1f}")
        
        print(f"\n🗄️ Cache Sizes:")
        for cache_type, size in stats['cache_sizes'].items():
            print(f"{cache_type.title():<18} {size:,}")
        
        print("="*60)
    
    def reset_statistics(self) -> None:
        """Reset performance counters."""
        self.hit_count = 0
        self.miss_count = 0
        

# Global cache instance for use across the codebase
_global_topology_cache: Optional[NetworkTopologyCache] = None

def get_topology_cache() -> NetworkTopologyCache:
    """Get the global topology cache instance."""
    global _global_topology_cache
    if _global_topology_cache is None:
        _global_topology_cache = NetworkTopologyCache()
    return _global_topology_cache

def set_topology_cache(cache: NetworkTopologyCache) -> None:
    """Set the global topology cache instance."""
    global _global_topology_cache
    _global_topology_cache = cache

def clear_topology_cache() -> None:
    """Clear the global topology cache."""
    global _global_topology_cache
    if _global_topology_cache is not None:
        _global_topology_cache.clear_cache()
        _global_topology_cache = None