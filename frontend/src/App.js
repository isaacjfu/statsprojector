import './App.css';
import {useState,useEffect} from 'react';
import Searchbar from './components/searchbar.js'

const backendURL = 'http://127.0.0.1:5000/'
function App() {
  const [players,setPlayers] = useState([])
  const [selectedPlayer, setSelectedPlayer] = useState("")
  const [isPlayerSelected,setIsPlayerSelected] = useState(false)

  useEffect ( () => {
    const playerFetch = async () => {
      try{
        const res = await fetch(backendURL)
        // Check if the response is ok (status code 200-299)
        if (!res.ok) {
          throw new Error(`HTTP error! Status: ${res.status}`);
        }
        const data = await res.json()
        setPlayers(data)
        console.log("in main" + players)
      } catch (error) {
        console.error(error)
      }
    }
    playerFetch()
}, [])
useEffect ( () => {
  console.log("selected player is " + selectedPlayer)
}, [selectedPlayer])
  return (
    <div className="App">
      <span>Choose a Player</span>
      <Searchbar players = {players} setSelectedPlayer = {setSelectedPlayer} setIsPlayerSelected = {setIsPlayerSelected} />
      {selectedPlayer && (
        <div className = "selected-player">
          <p>Selected Player: {selectedPlayer}</p>
        </div>
      )}
    </div>
  );
}

export default App;
