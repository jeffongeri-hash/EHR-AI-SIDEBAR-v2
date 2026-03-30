"""
Travel and Flight Search API Routes
"""

from typing import Dict, List, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.flight_service import flight_service
from loguru import logger

router = APIRouter(prefix="/api/travel", tags=["travel"])


class FlightSearchRequest(BaseModel):
    origin: str
    destination: str
    departure_date: str
    return_date: str = None
    class_type: str = "economy"
    passengers: int = 1


class RewardCalculationRequest(BaseModel):
    flight_price: float
    points_available: int
    airline: str = None


class RecommendationRequest(BaseModel):
    origin: str
    destination: str
    budget_points: int
    preferred_class: str = "business"
    flexible_dates: bool = True


@router.post("/search-flights")
async def search_flights(request: FlightSearchRequest) -> Dict[str, Any]:
    """Search for flights between two locations"""
    try:
        flights = await flight_service.search_flights(
            origin=request.origin,
            destination=request.destination,
            departure_date=request.departure_date,
            return_date=request.return_date,
            class_type=request.class_type,
            passengers=request.passengers
        )
        
        return {
            "success": True,
            "flights": [
                {
                    "id": f"{flight.airline}-{flight.flight_number}",
                    "airline": flight.airline,
                    "flight_number": flight.flight_number,
                    "origin": flight.origin,
                    "destination": flight.destination,
                    "departure_date": flight.departure_date,
                    "departure_time": flight.departure_time,
                    "arrival_time": flight.arrival_time,
                    "duration": flight.duration,
                    "price_usd": flight.price_usd,
                    "class_type": flight.class_type,
                    "stops": flight.stops,
                    "booking_url": flight.booking_url
                }
                for flight in flights
            ],
            "search_criteria": {
                "origin": request.origin,
                "destination": request.destination,
                "departure_date": request.departure_date,
                "return_date": request.return_date,
                "class_type": request.class_type,
                "passengers": request.passengers
            }
        }
    except Exception as e:
        logger.error(f"Flight search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calculate-rewards")
async def calculate_rewards(request: RewardCalculationRequest) -> Dict[str, Any]:
    """Calculate Chase Ultimate Rewards point value for a flight"""
    try:
        reward_value = flight_service.calculate_chase_rewards_value(
            points_available=request.points_available,
            flight_price=request.flight_price,
            airline=request.airline
        )
        
        return {
            "success": True,
            "reward_calculation": {
                "points_needed": reward_value.points_needed,
                "cash_equivalent": reward_value.cash_equivalent,
                "value_per_point": reward_value.value_per_point,
                "transfer_partners": reward_value.transfer_partners,
                "best_redemption": reward_value.best_redemption,
                "affordable": request.points_available >= reward_value.points_needed,
                "points_remaining": max(0, request.points_available - reward_value.points_needed),
                "points_needed_additional": max(0, reward_value.points_needed - request.points_available)
            },
            "flight_info": {
                "price_usd": request.flight_price,
                "airline": request.airline,
                "points_available": request.points_available
            }
        }
    except Exception as e:
        logger.error(f"Reward calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommendations")
async def get_flight_recommendations(request: RecommendationRequest) -> Dict[str, Any]:
    """Get personalized flight recommendations based on Chase points budget"""
    try:
        recommendations = await flight_service.get_flight_recommendations(
            origin=request.origin,
            destination=request.destination,
            budget_points=request.budget_points,
            preferred_class=request.preferred_class,
            flexible_dates=request.flexible_dates
        )
        
        # Format recommendations for API response
        formatted_recommendations = []
        for rec in recommendations["recommendations"]:
            flight = rec["flight"]
            rewards = rec["rewards_info"]
            
            formatted_recommendations.append({
                "flight": {
                    "id": f"{flight.airline}-{flight.flight_number}",
                    "airline": flight.airline,
                    "flight_number": flight.flight_number,
                    "origin": flight.origin,
                    "destination": flight.destination,
                    "departure_date": flight.departure_date,
                    "departure_time": flight.departure_time,
                    "arrival_time": flight.arrival_time,
                    "duration": flight.duration,
                    "price_usd": flight.price_usd,
                    "class_type": flight.class_type,
                    "stops": flight.stops,
                    "booking_url": flight.booking_url
                },
                "rewards": {
                    "points_needed": rewards.points_needed,
                    "cash_equivalent": rewards.cash_equivalent,
                    "value_per_point": rewards.value_per_point,
                    "best_redemption": rewards.best_redemption,
                    "transfer_partners": rewards.transfer_partners
                },
                "booking_info": {
                    "affordable": rec["affordable"],
                    "points_remaining": rec["points_remaining"],
                    "value_rating": "excellent" if rec["affordable"] and rewards.value_per_point > 0.015 
                                   else "good" if rec["affordable"] 
                                   else "over_budget"
                }
            })
        
        return {
            "success": True,
            "recommendations": formatted_recommendations,
            "budget_analysis": {
                "total_points": recommendations["budget_analysis"]["total_points"],
                "cash_equivalent": recommendations["budget_analysis"]["cash_equivalent"],
                "affordable_options": len(recommendations["budget_analysis"]["best_options"]),
                "best_value_option": formatted_recommendations[0] if formatted_recommendations else None
            },
            "search_criteria": {
                "origin": request.origin,
                "destination": request.destination,
                "budget_points": request.budget_points,
                "preferred_class": request.preferred_class,
                "flexible_dates": request.flexible_dates
            }
        }
    except Exception as e:
        logger.error(f"Flight recommendations failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/point-values")
async def get_chase_point_values() -> Dict[str, Any]:
    """Get current Chase Ultimate Rewards point values and transfer partners"""
    return {
        "success": True,
        "point_values": {
            "portal_booking": flight_service.chase_point_values["portal_booking"],
            "transfer_partners": flight_service.chase_point_values["transfer_partners"]
        },
        "recommendations": {
            "portal_booking": "Good for flexibility and simplicity",
            "transfer_partners": "Often provides better value for premium cabins",
            "best_general_value": "Transfer to partners for business/first class flights",
            "cash_equivalent": "Use portal for economy flights or when transfer partners don't offer better value"
        }
    }


@router.get("/airports")
async def search_airports(query: str = "") -> Dict[str, Any]:
    """Search for airports by code or city name"""
    # Mock airport data - in production this would query a real airport database
    airports = [
        {"code": "JFK", "name": "John F. Kennedy International", "city": "New York", "country": "USA"},
        {"code": "LAX", "name": "Los Angeles International", "city": "Los Angeles", "country": "USA"},
        {"code": "LHR", "name": "Heathrow", "city": "London", "country": "UK"},
        {"code": "NRT", "name": "Narita International", "city": "Tokyo", "country": "Japan"},
        {"code": "CDG", "name": "Charles de Gaulle", "city": "Paris", "country": "France"},
        {"code": "FRA", "name": "Frankfurt", "city": "Frankfurt", "country": "Germany"},
        {"code": "SIN", "name": "Singapore Changi", "city": "Singapore", "country": "Singapore"},
        {"code": "DXB", "name": "Dubai International", "city": "Dubai", "country": "UAE"},
        {"code": "SYD", "name": "Sydney Kingsford Smith", "city": "Sydney", "country": "Australia"},
        {"code": "ORD", "name": "O'Hare International", "city": "Chicago", "country": "USA"},
    ]
    
    if query:
        query_lower = query.lower()
        filtered_airports = [
            airport for airport in airports 
            if query_lower in airport["code"].lower() 
            or query_lower in airport["city"].lower()
            or query_lower in airport["name"].lower()
        ]
    else:
        filtered_airports = airports
    
    return {
        "success": True,
        "airports": filtered_airports,
        "query": query
    }