import {useState, useEffect} from 'react'

const Searchbar = ( {players, setSelectedPlayer} ) => {
    const [searchTerm, setSearchTerm] = useState('');
    const [filteredPlayers, setFilteredPlayers] = useState([]);
    const handleSearchChange = (e) => {
        const value = e.target.value
        setSearchTerm(value)
        if(value){
            setFilteredPlayers( players.filter((player) =>
                player.full_name.toLowerCase().includes(value.toLowerCase())
            ));
        } else{
            setFilteredPlayers([]);
        }
    }

    const handleSelectPlayer = (player) => {
        setSearchTerm(player.full_name)
        setSelectedPlayer(player.full_name)
        setFilteredPlayers([])
    }
    return (
        <div className="search-container">
        <input
          type="text"
          value={searchTerm}
          onChange={handleSearchChange}
          placeholder="Search players..."
          className="search-input"
        />
        
        {filteredPlayers.length > 0 && (
          <ul className="dropdown-list">
            {filteredPlayers.map((player) => (
              <li 
                key={player.id} 
                onClick={() => handleSelectPlayer(player)} 
                className="dropdown-item"
              >
                {player.full_name}
              </li>
            ))}
          </ul>
        )}
  
      </div>
    )

}

export default Searchbar