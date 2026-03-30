"""
Google Flights API Integration Service
Provides flight search and booking capabilities
"""

import asyncio
import httpx
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger
import json

@dataclass
class FlightResult:
    """Flight search result"""
    origin: str
    destination: str
    departure_date: str
    return_date: Optional[str]
    airline: str
    flight_number: str
    duration: str
    price_usd: float
    class_type: str
    stops: int
    departure_time: str
    arrival_time: str
    booking_url: Optional[str] = None

@dataclass
class ChaseRewardValue:
    """Chase Ultimate Rewards value calculation"""
    points_needed: int
    cash_equivalent: float
    value_per_point: float
    transfer_partners: List[str]
    best_redemption: str

class FlightService:
    """Service for flight search and booking integration"""
    
    def __init__(self):
        self.base_url = "https://serpapi.com/search"
        # Note: In production, use environment variables for API keys
        self.api_key = None  # Set via environment variable
        
        # Chase Ultimate Rewards point values (typical valuations)
        self.chase_point_values = {
            "portal_booking": 0.0125,  # 1.25 cents per point through Chase portal
            "transfer_partners": {
                "United": 0.015,
                "Southwest": 0.014,
                "British Airways": 0.016,
                "Air France/KLM": 0.014,
                "Singapore": 0.018,
                "Hyatt": 0.019
            }
        }
    
    async def search_flights(
        self,
        origin: str,
        destination: str,
        departure_date: str,
        return_date: Optional[str] = None,
        class_type: str = "economy",
        passengers: int = 1
    ) -> List[FlightResult]:
        """Search for flights using Google Flights via SerpAPI"""
        
        try:
            if not self.api_key:
                # Return mock data for development
                return await self._get_mock_flights(origin, destination, departure_date, class_type)
            
            params = {
                "engine": "google_flights",
                "departure_id": origin,
                "arrival_id": destination,
                "outbound_date": departure_date,
                "return_date": return_date,
                "currency": "USD",
                "hl": "en",
                "api_key": self.api_key,
                "travel_class": self._map_class_type(class_type),
                "adults": passengers
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()
                
                data = response.json()
                return self._parse_flight_results(data)
                
        except Exception as e:
            logger.error(f"Flight search failed: {e}")
            # Return mock data as fallback
            return await self._get_mock_flights(origin, destination, departure_date, class_type)
    
    async def _get_mock_flights(
        self, 
        origin: str, 
        destination: str, 
        departure_date: str,
        class_type: str
    ) -> List[FlightResult]:
        """Generate mock flight data for development/demo"""
        
        # Simulate API delay
        await asyncio.sleep(1)
        
        base_prices = {
            "economy": 800,
            "business": 3500,
            "first": 8500
        }
        
        airlines = ["United", "American", "Delta", "Lufthansa", "British Airways", "Singapore"]
        
        flights = []
        for i, airline in enumerate(airlines[:3]):
            price_multiplier = 1 + (i * 0.15)  # Vary prices
            base_price = base_prices.get(class_type.lower(), 800)
            
            flights.append(FlightResult(
                origin=origin,
                destination=destination,
                departure_date=departure_date,
                return_date=None,
                airline=airline,
                flight_number=f"{airline[:2].upper()}{1000 + i}",
                duration=f"{12 + i}h {30 + (i*15)}m",
                price_usd=int(base_price * price_multiplier),
                class_type=class_type,
                stops=i,
                departure_time=f"{8 + i}:{30 + (i*15):02d} AM",
                arrival_time=f"{10 + i}:{45 + (i*20):02d} PM",
                booking_url=f"https://booking.{airline.lower()}.com/flights/{origin}-{destination}"
            ))
        
        return flights
    
    def calculate_chase_rewards_value(
        self, 
        points_available: int, 
        flight_price: float,
        airline: str = None
    ) -> ChaseRewardValue:
        """Calculate best Chase Ultimate Rewards redemption for a flight"""
        
        # Portal booking value
        portal_points_needed = int(flight_price / self.chase_point_values["portal_booking"])
        
        # Transfer partner value (if applicable)
        best_transfer_value = 0.0125  # Default to portal value
        best_partner = "Chase Portal"
        
        if airline and airline in self.chase_point_values["transfer_partners"]:
            transfer_value = self.chase_point_values["transfer_partners"][airline]
            transfer_points_needed = int(flight_price / transfer_value)
            
            if transfer_points_needed < portal_points_needed:
                best_transfer_value = transfer_value
                best_partner = f"Transfer to {airline}"
                portal_points_needed = transfer_points_needed
        
        # Check all transfer partners for best value
        for partner, value in self.chase_point_values["transfer_partners"].items():
            partner_points_needed = int(flight_price / value)
            if partner_points_needed < portal_points_needed:
                portal_points_needed = partner_points_needed
                best_transfer_value = value
                best_partner = f"Transfer to {partner}"
        
        return ChaseRewardValue(
            points_needed=portal_points_needed,
            cash_equivalent=flight_price,
            value_per_point=best_transfer_value,
            transfer_partners=list(self.chase_point_values["transfer_partners"].keys()),
            best_redemption=best_partner
        )
    
    def _map_class_type(self, class_type: str) -> int:
        """Map class type to Google Flights API format"""
        mapping = {
            "economy": 1,
            "premium_economy": 2,
            "business": 3,
            "first": 4
        }
        return mapping.get(class_type.lower(), 1)
    
    def _parse_flight_results(self, data: Dict[str, Any]) -> List[FlightResult]:
        """Parse Google Flights API response"""
        flights = []
        
        try:
            if "best_flights" in data:
                for flight_data in data["best_flights"]:
                    flights.append(self._parse_single_flight(flight_data))
            
            if "other_flights" in data:
                for flight_data in data["other_flights"]:
                    flights.append(self._parse_single_flight(flight_data))
                    
        except Exception as e:
            logger.error(f"Error parsing flight results: {e}")
        
        return flights
    
    def _parse_single_flight(self, flight_data: Dict[str, Any]) -> FlightResult:
        """Parse a single flight from API response"""
        
        # Extract flight information from API response structure
        flights_info = flight_data.get("flights", [{}])[0]
        
        return FlightResult(
            origin=flights_info.get("departure_airport", {}).get("id", ""),
            destination=flights_info.get("arrival_airport", {}).get("id", ""),
            departure_date=flights_info.get("departure_airport", {}).get("time", ""),
            return_date=None,
            airline=flights_info.get("airline", "Unknown"),
            flight_number=flights_info.get("flight_number", ""),
            duration=flight_data.get("total_duration", ""),
            price_usd=flight_data.get("price", 0),
            class_type=flights_info.get("travel_class", "economy"),
            stops=len(flight_data.get("layovers", [])),
            departure_time=flights_info.get("departure_airport", {}).get("time", ""),
            arrival_time=flights_info.get("arrival_airport", {}).get("time", ""),
            booking_url=flight_data.get("booking_url")
        )

    async def get_flight_recommendations(
        self,
        origin: str,
        destination: str,
        budget_points: int,
        preferred_class: str = "business",
        flexible_dates: bool = True
    ) -> Dict[str, Any]:
        """Get personalized flight recommendations based on Chase points budget"""
        
        # Search for flights
        departure_date = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        flights = await self.search_flights(
            origin=origin,
            destination=destination, 
            departure_date=departure_date,
            class_type=preferred_class
        )
        
        recommendations = []
        for flight in flights:
            reward_value = self.calculate_chase_rewards_value(
                budget_points, 
                flight.price_usd, 
                flight.airline
            )
            
            # Check if user has enough points
            affordable = budget_points >= reward_value.points_needed
            
            recommendations.append({
                "flight": flight,
                "rewards_info": reward_value,
                "affordable": affordable,
                "points_remaining": budget_points - reward_value.points_needed if affordable else 0
            })
        
        # Sort by value and affordability
        recommendations.sort(key=lambda x: (x["affordable"], -x["rewards_info"].value_per_point))
        
        return {
            "recommendations": recommendations,
            "budget_analysis": {
                "total_points": budget_points,
                "cash_equivalent": budget_points * 0.0125,
                "best_options": [r for r in recommendations if r["affordable"]][:3]
            }
        }

# Global service instance
flight_service = FlightService()