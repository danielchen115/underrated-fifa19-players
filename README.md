# Finding the most underrated players in FIFA 19
It's difficult to put a price on a player. There's so many factors that go into play to decide how "good" a player is, 
and it's hard (if not impossible) to get it 100% correct. I want to find out which players give the most bang for their buck; The players that should be worth way more than what they're listed as.

## Approach
EA assigns metrics to each of their players based on different factors of a player. This includes stuff like _passing_, _finishing_, _dribbling_, etc.
An underrated player will have metrics similar to their counterparts, except lower in price. 
First, I find their predicted value using a regression model. Then, I see the difference between their predicted value with their actual value. Those with a significantly higher predicted value are _underrated_.

## Next Steps
* Divide players by position. Though the machine learning model implicitly does this, it'd be nice to see how it compares.
* Discover "styles" of players with k-means clustering. It'd be interesting to see what type of player archetypes are common in the game.

